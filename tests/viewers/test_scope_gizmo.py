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
    face_axis,
    ray_hit,
    rebased,
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
    """On an axis the box has collapsed the two grips are the SAME
    square, so a ray perpendicular to it hits both at the same distance.
    Whichever one wins, it must be the same one every time — and it is
    the earliest in ``FACE_HANDLES``."""
    flat = _box(lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 0.0))   # a Z-slab
    geom = scope_gizmo_geometry(_GID, flat, _BBOX)
    origin, direction = (0.0, 0.0, 10.0), (0.0, 0.0, -1.0)
    first = ray_hit(geom, origin, direction)
    assert first is not None
    assert first == ("zmin", pytest.approx(10.0))   # zmin before zmax
    for _ in range(5):
        assert ray_hit(geom, origin, direction) == first


def test_a_corner_is_not_a_handle():
    """The old gizmo made the six FULL faces the grab area, so a ray
    anywhere into the box hit one — including straight at a corner.

    That is the F1 defect in miniature: the box seeds to the geometry's
    own bounds, so "anywhere into the box" is "anywhere over the model".
    A corner is not a handle; only the six grips are.
    """
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    direction = tuple(-np.ones(3) / math.sqrt(3.0))
    assert ray_hit(geom, (10.0, 10.0, 10.0), direction) is None


def test_the_grip_is_sized_from_the_model_not_from_the_box():
    """A box collapsed to a single POINT has no face with any area, and
    a grab region derived from the box would therefore have none either
    — a dead end the user could only escape through the panel.

    The ray is deliberately fired OFF the collapsed point: aiming
    straight down its centre would pass the containment test even with
    a zero-sized grip, which is what made the previous version of this
    test vacuous (review finding F4 — deleting the sizing left it
    green). Here the hit depends on the grip's own extent and nothing
    else, so it is exactly what fails if the grip ever starts being
    sized from the box.
    """
    point = _box(lo=(0.0, 0.0, 0.0), hi=(0.0, 0.0, 0.0))
    geom = scope_gizmo_geometry(_GID, point, _BBOX)
    half = geom.grip_half
    assert half > 0.0

    # Inside the grip but far outside the (zero-extent) box.
    inside = ray_hit(geom, (0.6 * half, 0.6 * half, 10.0), (0.0, 0.0, -1.0))
    assert inside is not None and inside[0] == "zmin"
    # …and the grip is BOUNDED: past its corner there is nothing to grab.
    assert ray_hit(
        geom, (1.5 * half, 0.0, 10.0), (0.0, 0.0, -1.0),
    ) is None


def test_the_grab_region_is_exactly_the_drawn_grip():
    """F8: the old hit test widened a face's containment by a floor
    applied unconditionally, so on a box 0.005 thick the perpendicular
    faces stayed grabbable ±0.069 out — ~14× the drawn thickness, i.e.
    a click plainly outside the visible box started a drag.

    There is now nothing to bound, because there is no widening: the
    hit test and the layer builder read the same four corners.
    """
    thin = _box(lo=(-1.0, -1.0, -0.005), hi=(1.0, 1.0, 0.005))
    geom = scope_gizmo_geometry(_GID, thin, _BBOX)

    for handle in FACE_HANDLES:
        quad = geom.grip_quad(handle)
        axis = face_axis(handle)
        others = [a for a in (0, 1, 2) if a != axis]
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        # The grip is flat in its face's plane, at the bound it moves.
        assert lo[axis] == pytest.approx(hi[axis])
        assert lo[axis] == pytest.approx(geom.bound(handle))

        # Fire perpendicular to the grip, just inside and just outside
        # each of its own drawn corners.
        direction = np.zeros(3)
        direction[axis] = -1.0
        for other in others:
            for frac, expect_hit in ((0.98, True), (1.02, False)):
                origin = np.asarray(geom.face_centre(handle), dtype=float)
                origin[other] += frac * geom.grip_half
                origin[axis] += 10.0
                hit = ray_hit(geom, origin, direction)
                assert (hit is not None) is expect_hit, (
                    f"{handle}: {frac:.2f} of the drawn half-extent "
                    f"along axis {other} should "
                    f"{'hit' if expect_hit else 'miss'}"
                )


# =====================================================================
# S-MISS / install — the priority-12 abort discipline
# =====================================================================

def test_a_miss_does_not_abort():
    backend, _s, _r, _it, _f = _rig()
    assert backend.iren.send("LeftButtonPressEvent", (5.0, 5.0)) is False


def test_a_press_at_the_model_centroid_is_left_to_the_picker():
    """F1, the critical one: enabling the scope seeds the box to the
    geometry's OWN bounds, so the box encloses the model.

    With the six full faces as handles, every ray entering the box hit
    one — so every press over the model silhouette became a face drag
    and was aborted at priority 12. The picker at priority 10 never saw
    it: node pick, element pick and rubber-band box-pick all died
    silently, with a SIZEALL cursor everywhere. A press in the middle of
    the model must reach the picker.
    """
    backend, _s, renderer, interactor, _f = _rig()
    geom = _geom(renderer)
    centroid = (np.asarray(geom.lo) + np.asarray(geom.hi)) / 2.0
    pixel = backend.pixel_for(centroid)

    assert interactor.hit(*pixel) == (None, None)
    assert backend.iren.send("LeftButtonPressEvent", pixel) is False, (
        "the scope gizmo aborted a press over the model interior — the "
        "picker at priority 10 will never see it"
    )
    assert not interactor.is_dragging


def test_a_model_sized_box_claims_almost_none_of_the_model():
    """The same defect, measured rather than sampled at one point.

    Every pixel whose ray enters the box used to be claimed — 100 % of
    the model's silhouette, which is what the review measured against a
    real ``ResultsViewer`` (27/27 sampled interior pixels). The grips
    are a fixed small square each, so the claimed set is now ~5 % of
    this lattice, and the rest of the model stays pickable.

    Not zero, and it cannot be: a grip sits inside its face, so the
    pixels showing it are by definition pixels over the model. The
    contract is that they are a small minority rather than all of them.
    """
    backend, _s, renderer, interactor, _f = _rig()
    geom = _geom(renderer)
    lo, hi = np.asarray(geom.lo), np.asarray(geom.hi)

    axis = np.linspace(0.0, 1.0, 7)
    interior = [
        lo + (hi - lo) * np.array([fx, fy, fz])
        for fx in axis for fy in axis for fz in axis
    ]
    claimed = sum(
        interactor.hit(*backend.pixel_for(p))[1] is not None
        for p in interior
    )
    assert claimed / len(interior) < 0.10, (
        f"{claimed}/{len(interior)} interior points claimed — the gizmo "
        f"is eating the model"
    )
    assert claimed > 0, "…but the grips must still be reachable"


def test_a_grip_stays_reachable_on_a_box_far_bigger_than_the_model():
    """The other half of the F1 bargain: a grab area that does not grow
    with the box must not become unhittable when the box is huge."""
    huge = _box(lo=(-40.0, -40.0, -40.0), hi=(40.0, 40.0, 40.0))
    backend, _s, renderer, interactor, _f = _rig(box=huge)
    for handle in FACE_HANDLES:
        pixel = backend.pixel_for(_geom(renderer).face_centre(handle))
        assert interactor.hit(*pixel)[0] == _GID, (
            f"{handle}'s grip is unreachable on a 80-unit box"
        )


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


def test_uninstall_withdraws_the_cursor_it_asked_for():
    """F10: the arbiter holds a memo per requester, so an interactor
    torn down mid-hover left a SIZEALL request standing that nothing
    could ever retract — the window kept the drag cursor over a gizmo
    that no longer exists."""
    import vtk

    backend, _s, renderer, interactor, _f = _rig()
    window = backend.render_window
    pixel = backend.pixel_for(_geom(renderer).face_centre("zmax"))

    backend.iren.send("MouseMoveEvent", pixel)      # hover a grip
    assert window.cursor == vtk.VTK_CURSOR_SIZEALL

    interactor.uninstall()
    assert window.cursor == vtk.VTK_CURSOR_DEFAULT, (
        "the torn-down interactor left its cursor request standing"
    )
    # …and it does not hold the interactor alive afterwards.
    assert interactor._iren is None
    interactor.uninstall()                          # idempotent


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

def _mask_recomputes(interactor) -> "tuple[list, object]":
    """A stand-in for ``ResultsViewer._on_geometry_scope_changed``.

    Same gate the real subscriber uses — ``is_dragging`` — so what this
    counts is the contract, not a re-implementation of it.
    """
    runs: list = []

    def recompute(geometry_id):
        if interactor.is_dragging:
            return
        runs.append(geometry_id)

    return runs, recompute


def test_a_drag_defers_the_mask_to_the_release():
    """F2: the mask pass is O(cells) per materialized geometry plus a
    ghost recompose — measured at 2.5 ms @50k cells, 11.8 ms @220k and
    28.1 ms @500k — and it used to run on EVERY mouse-move, before the
    render.

    The clip gizmo had already solved this; ``is_dragging`` existed here
    but nothing consulted it. Twenty moves must now cost ONE recompute,
    and the box must still land exactly where the cursor left it.
    """
    backend, scopes, renderer, interactor, fired = _rig()
    runs, recompute = _mask_recomputes(interactor)
    interactor.on_changed = lambda gid: (fired.append(gid), recompute(gid))
    interactor.on_drag_end = recompute

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
    assert runs == [], "the mask recomputed mid-drag"
    backend.iren.send("LeftButtonReleaseEvent")

    # Exactly one mask pass, on the release…
    assert runs == [_GID]
    # …and the VALUE it lands on is the one the cursor asked for: the
    # face started at x = 1.0 and the last move was +0.40.
    moved = scopes.box_for(_GID)
    assert float(moved.max[0]) == pytest.approx(1.40)
    assert list(moved.min) == pytest.approx([-1.0, -1.0, -1.0])
    assert list(moved.max[1:]) == pytest.approx([1.0, 1.0])

    # The gizmo and the panel still followed every tick — only the mask
    # waited — and the layers were reused in place, so a drag adds and
    # removes no actors.
    assert fired == [_GID] * 20
    kinds = [c[0] for c in backend.calls]
    assert set(kinds) == {"ray", "update"}
    assert "add" not in kinds and "remove" not in kinds
    assert kinds.count("update") == 40          # grips + edges, 20 moves


def test_a_drag_end_fires_once_with_the_gesture_already_settled():
    """The release half of the deal. ``is_dragging`` must already read
    False inside the callback, or the pass it exists to trigger would
    gate itself out."""
    backend, _s, renderer, interactor, _f = _rig()
    seen: list = []
    interactor.on_drag_end = lambda gid: seen.append(
        (gid, interactor.is_dragging),
    )
    start = np.asarray(_geom(renderer).face_centre("zmax"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.2])),
    )
    backend.iren.send("LeftButtonReleaseEvent")
    assert seen == [(_GID, False)]

    # A second release with nothing in flight announces nothing.
    backend.iren.send("LeftButtonReleaseEvent")
    assert seen == [(_GID, False)]


def test_a_leave_ends_the_drag_through_the_same_boundary():
    """A release delivered outside the window never arrives, so the
    deferred pass has to hang off the LeaveEvent too — otherwise a drag
    that ends off-window leaves the mask stale forever."""
    backend, _s, renderer, interactor, _f = _rig()
    seen: list = []
    interactor.on_drag_end = seen.append
    start = np.asarray(_geom(renderer).face_centre("zmax"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.2])),
    )
    backend.iren.send("LeaveEvent")
    assert seen == [_GID]


def test_a_move_that_rebuilds_the_same_box_announces_nothing():
    """F2's other half: a move that changes no bound must not write the
    owner or fire the mask event.

    Qt delivers a ``MouseMoveEvent`` for sub-pixel motion, so a hand
    resting on a pressed button produces a stream of moves at an
    unchanged pixel — each of which used to write the identical box and
    re-run the O(cells) pass over every geometry to change nothing.
    """
    backend, scopes, renderer, _it, fired = _rig()
    start = np.asarray(_geom(renderer).face_centre("zmax"))
    press = backend.pixel_for(start)

    backend.iren.send("LeftButtonPressEvent", press)
    # The cursor has not moved off the press pixel yet.
    for _ in range(4):
        assert backend.iren.send("MouseMoveEvent", press) is True
    assert scopes.set_calls == [], "a no-op move wrote the owner"
    assert fired == [], "a no-op move fired the mask event"

    # A move that DOES change the box is announced, exactly once…
    moved = backend.pixel_for(start + np.array([0.0, 0.0, 0.3]))
    backend.iren.send("MouseMoveEvent", moved)
    assert len(scopes.set_calls) == 1 and fired == [_GID]
    assert float(scopes.box_for(_GID).max[2]) == pytest.approx(1.3)

    # …and holding still there is quiet again.
    for _ in range(4):
        backend.iren.send("MouseMoveEvent", moved)
    assert len(scopes.set_calls) == 1
    assert fired == [_GID]


def test_an_external_write_mid_drag_is_not_reverted():
    """F11: the press froze all six bounds, so the next mouse-move
    rebuilt the box from the press-time snapshot and silently threw
    away anything the panel (or a session restore) had written in
    between. Only the axis ANCHOR needs freezing."""
    backend, scopes, renderer, _it, _f = _rig()
    start = np.asarray(_geom(renderer).face_centre("zmax"))
    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))

    # Someone else narrows Y while the drag is in flight.
    scopes.set_scope(_GID, _box(lo=(-1.0, -0.25, -1.0), hi=(1.0, 0.25, 1.0)))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.3])),
    )

    box = scopes.box_for(_GID)
    assert float(box.max[2]) == pytest.approx(1.3)      # the drag landed
    assert list(box.min) == pytest.approx([-1.0, -0.25, -1.0]), (
        "the drag reverted an external write to Y"
    )
    assert list(box.max[:2]) == pytest.approx([1.0, 0.25])


def test_rebased_keeps_the_grip_size_and_takes_the_new_bounds():
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    moved = rebased(geom, _box(lo=(0.0, 0.0, 0.0), hi=(3.0, 3.0, 3.0)))
    assert moved.lo == pytest.approx((0.0, 0.0, 0.0))
    assert moved.hi == pytest.approx((3.0, 3.0, 3.0))
    assert moved.grip_half == geom.grip_half
    assert moved.geometry_id == geom.geometry_id


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
        f"{SCOPE_GIZMO_LAYER_PREFIX}g1:grips",
        f"{SCOPE_GIZMO_LAYER_PREFIX}g1:edges",
    }
    assert set(renderer.geometries()) == {"g1"}

    active[0] = "g2"
    renderer.refresh()
    assert _layer_ids(backend) == {
        f"{SCOPE_GIZMO_LAYER_PREFIX}g2:grips",
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
    edges = backend.layers[f"{SCOPE_GIZMO_LAYER_PREFIX}{_GID}:edges"]
    pts = np.asarray(edges.points.coords)
    assert pts.shape == (8, 3)
    assert pts.min(axis=0) == pytest.approx(list(box.min))
    assert pts.max(axis=0) == pytest.approx(list(box.max))


def test_the_drawn_grips_are_the_grab_surface():
    """The grips layer paints the same six quads :func:`ray_hit` tests
    — corner for corner, so what you grab is what you see (F8)."""
    backend, _s, renderer, _it, _f = _rig()
    grips = backend.layers[f"{SCOPE_GIZMO_LAYER_PREFIX}{_GID}:grips"]
    pts = np.asarray(grips.points.coords)
    assert pts.shape == (24, 3)             # 6 grips x 4 corners
    quads = np.asarray(grips.cells.blocks["quad"])
    assert quads.shape == (6, 4)

    geom = _geom(renderer)
    for i, handle in enumerate(FACE_HANDLES):
        assert pts[quads[i]] == pytest.approx(geom.grip_quad(handle))


def test_a_fully_collapsed_box_still_draws_its_grips():
    """F3: collapsed on all three axes the 8 corners coincide, so the
    cage is 12 zero-length lines and draws NOTHING — while the scope
    hides every cell, leaving an entirely empty viewport with no cue
    and no way back. Three ordinary drags reach that state.

    The grips are not sized from the box, so they are still there: full
    size, at the point, and still grabbable.
    """
    point = _box(lo=(0.5, 0.5, 0.5), hi=(0.5, 0.5, 0.5))
    backend, scopes, renderer, interactor, _f = _rig(box=point)

    edges = backend.layers[f"{SCOPE_GIZMO_LAYER_PREFIX}{_GID}:edges"]
    cage = np.asarray(edges.points.coords)
    assert cage.min(axis=0) == pytest.approx(cage.max(axis=0)), (
        "only meaningful while the cage really has collapsed to a point"
    )

    grips = backend.layers[f"{SCOPE_GIZMO_LAYER_PREFIX}{_GID}:grips"]
    pts = np.asarray(grips.points.coords)
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert bool(np.all(extent > 0.0)), "nothing at all is drawn"
    assert extent == pytest.approx([2 * _geom(renderer).grip_half] * 3)

    # …and it is recoverable with the mouse, not only from the panel.
    # Every grip is centred on the same point, so a ray through it ties
    # across all six and the declared order names ``xmin``; the box
    # therefore reopens along X (``resolve_side`` lets a coincident pair
    # take whichever bound the motion implies).
    pixel = backend.pixel_for(np.array([0.5, 0.5, 0.5]))
    assert interactor.hit(*pixel) == (_GID, "xmin")
    assert backend.iren.send("LeftButtonPressEvent", pixel) is True
    backend.iren.send(
        "MouseMoveEvent", backend.pixel_for(np.array([0.9, 0.5, 0.5])),
    )
    reopened = scopes.box_for(_GID)
    assert float(reopened.max[0]) - float(reopened.min[0]) == pytest.approx(
        0.4, abs=1e-6,
    )


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
def test_a_real_press_over_the_model_still_reaches_the_picker(cube_results):
    """F1 end to end, through a real VTK interactor.

    The review measured this against a real ``ResultsViewer``: with the
    scope enabled at the geometry's own bounds, 27/27 sampled interior
    pixels were claimed by the gizmo, and ``set_show_gizmo(False)``
    made the same pixel return ``(None, None)``. The gizmo was eating
    every pick over the model.

    The proof here is the pick backend's own state: it records the
    press position at priority 10, so if the priority-12 gizmo aborts,
    ``_press_pos`` stays ``None`` and the entire click-pick and
    rubber-band gesture is dead before it starts.
    """
    _need_qt()
    from qtpy import QtCore, QtWidgets

    from apeGmsh.viewers.diagrams._dispatch import GEOMETRY_SCOPE_CHANGED
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="scope-gizmo-pick",
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
            # Exactly what enabling the scope does: seed to the
            # geometry's own bounds, so the box encloses the model.
            director.scopes.set_scope(geom.id, box)
            director.dispatcher.fire(
                GEOMETRY_SCOPE_CHANGED, payload=geom.id,
            )
            QtWidgets.QApplication.processEvents()
            plotter.render()

            interactor = viewer._scope_gizmo_interactor
            gizmo = viewer._scope_gizmos.geometries()[geom.id]
            centroid = np.asarray(box.center)
            px = _to_pixel(plotter, centroid)
            seen["centroid_hit"] = interactor.hit(*px)

            pick_backend = viewer._pick_controller._backend
            picks: list = []
            original = pick_backend._on_pick
            pick_backend._on_pick = lambda hit, mods: (
                picks.append(hit), original(hit, mods),
            )

            iren = plotter.iren.interactor
            for event in (
                "LeftButtonPressEvent", "LeftButtonReleaseEvent",
            ):
                iren.SetEventPosition(
                    int(round(px[0])), int(round(px[1])),
                )
                iren.InvokeEvent(event)
                if event == "LeftButtonPressEvent":
                    seen["press_reached_picker"] = (
                        pick_backend._press_pos is not None
                    )
            seen["picked"] = len(picks)
            seen["picked_an_actor"] = bool(
                picks and picks[0] is not None
                and picks[0].prop_id is not None
            )
            seen["dragging"] = interactor.is_dragging

            # …while a press ON a grip is still claimed, so the gizmo
            # has not simply been switched off.
            grip = _to_pixel(plotter, gizmo.face_centre("zmax"))
            seen["grip_hit"] = interactor.hit(*grip)[1]
            pick_backend._press_pos = None
            iren.SetEventPosition(
                int(round(grip[0])), int(round(grip[1])),
            )
            iren.InvokeEvent("LeftButtonPressEvent")
            seen["grip_claimed"] = pick_backend._press_pos is None
            iren.InvokeEvent("LeftButtonReleaseEvent")
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(600, _drive_then_close)
    viewer.show()

    assert seen.get("centroid_hit") == (None, None), (
        "the gizmo claims the centre of the model"
    )
    assert seen.get("press_reached_picker") is True, (
        "the priority-12 gizmo aborted a press over the model — the "
        "picker at priority 10 never saw it"
    )
    assert seen["picked"] == 1
    assert seen["picked_an_actor"] is True, (
        "the click reached the picker but resolved nothing — the ray "
        "should have hit the model at its own centroid"
    )
    assert seen["dragging"] is False
    # …and the grips themselves are still live.
    assert seen["grip_hit"] == "zmax"
    assert seen["grip_claimed"] is True


@pytest.mark.qt
def test_a_real_drag_recomputes_the_mask_once(cube_results):
    """F2 end to end: the mask pass is O(cells) per materialized
    geometry and used to run on every mouse-move, synchronously, before
    the render (28 ms at 500k cells).

    Ten moves, one recompute — on the release — and the box still lands
    exactly where the cursor left it.
    """
    _need_qt()
    from qtpy import QtCore, QtWidgets

    from apeGmsh.viewers.core import scope_controller as sc
    from apeGmsh.viewers.diagrams._dispatch import GEOMETRY_SCOPE_CHANGED
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="scope-gizmo-defer",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    # Count the real O(cells) work, not a method call: ``refresh``
    # reaches this for every geometry it masks. ``ScopeController`` is
    # ``__slots__``-ed, so the module function is the seam.
    runs: list = []
    original = sc.compute_hidden_mask

    def _counted(grid, points, box):
        runs.append(int(getattr(grid, "n_cells", 0)))
        return original(grid, points, box)

    sc.compute_hidden_mask = _counted

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
            QtWidgets.QApplication.processEvents()

            span = float(box.max[2] - box.min[2])
            centre = box.center
            plotter.camera_position = [
                tuple(centre + np.array([6.0, 5.0, 4.0]) * span),
                tuple(centre),
                (0.0, 0.0, 1.0),
            ]
            plotter.render()

            gizmo = viewer._scope_gizmos.geometries()[geom.id]
            face = np.asarray(gizmo.face_centre("zmax"))
            delta = 0.25 * span
            px0 = _to_pixel(plotter, face)
            seen["hit"] = viewer._scope_gizmo_interactor.hit(*px0)

            # Enabling the scope above legitimately ran one pass; the
            # count under test starts at the press.
            seen["runs_on_enable"] = len(runs)
            runs.clear()

            iren = plotter.iren.interactor
            iren.SetEventPosition(int(round(px0[0])), int(round(px0[1])))
            iren.InvokeEvent("LeftButtonPressEvent")

            steps = 10
            for i in range(1, steps + 1):
                target = face + np.array([0.0, 0.0, delta * i / steps])
                px = _to_pixel(plotter, target)
                iren.SetEventPosition(int(round(px[0])), int(round(px[1])))
                iren.InvokeEvent("MouseMoveEvent")
                QtWidgets.QApplication.processEvents()
            seen["runs_during_drag"] = list(runs)
            seen["moved_during_drag"] = float(
                director.scopes.box_for(geom.id).max[2]
            )

            iren.InvokeEvent("LeftButtonReleaseEvent")
            QtWidgets.QApplication.processEvents()
            seen["runs_total"] = list(runs)

            moved = director.scopes.box_for(geom.id)
            seen["moved_max"] = float(moved.max[2])
            seen["expected"] = float(box.max[2]) + delta
            seen["span"] = span
            seen["moved_min"] = list(moved.min)
            seen["moved_max_xy"] = list(moved.max[:2])
            seen["start_min"] = list(box.min)
            seen["start_max_xy"] = list(box.max[:2])
            seen["start_max_z"] = float(box.max[2])
        finally:
            sc.compute_hidden_mask = original
            viewer._win.window.close()

    QtCore.QTimer.singleShot(600, _drive_then_close)
    viewer.show()

    assert seen.get("hit", (None, None))[1] == "zmax", (
        "the test only means something while the press grabs zmax"
    )
    # Ten moves, and the O(cells) pass ran for none of them…
    assert seen["runs_during_drag"] == [], (
        f"the mask recomputed {len(seen['runs_during_drag'])} times "
        f"mid-drag"
    )
    assert seen["runs_on_enable"] >= 1, (
        "enabling the scope must still recompute — only a DRAG defers"
    )
    # …but the BOX followed the cursor the whole way, so the gizmo and
    # the panel stayed live mid-drag; it is only the mask that waited.
    assert seen["moved_during_drag"] > seen["start_max_z"]
    # …and exactly one pass on the release.
    assert len(seen["runs_total"]) == 1, (
        f"expected one mask pass on release, got {seen['runs_total']}"
    )

    # The VALUE, not just the counts: the bound is where the cursor left
    # it, and the other five did not move.
    tol = 0.08 * seen["span"]
    assert seen["moved_max"] == pytest.approx(seen["expected"], abs=tol)
    assert seen["moved_min"] == pytest.approx(seen["start_min"])
    assert seen["moved_max_xy"] == pytest.approx(seen["start_max_xy"])


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
            # The TEXT, not ``value()``: with keyboard tracking off
            # (the fix that makes one edit one owner mutation) an
            # uncommitted edit lives in the line edit and ``value()``
            # still reports the last committed number. What must survive
            # a refresh is the digits the user can see.
            seen["typed_still_there"] = str(sb.text())
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
    assert "99" in seen["typed_still_there"], (
        f"the refresh ate the keystrokes (ADR 0083 B3): the field reads "
        f"{seen['typed_still_there']!r}"
    )
    assert seen["still_focused"] is True
    assert seen["unfocused_followed"] == pytest.approx(
        seen["before"], abs=1e-6,
    )
