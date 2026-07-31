"""Section-plane gizmos — the handle in the picture (ADR 0083 S2).

Same discipline as ``test_clip_planes.py`` (assert pixels and
behaviour, not that VTK was called) plus the ``test_legend_interactor``
rig style (a scriptable fake interactor drives the priority-12
observers):

* **G-RENDER** — gizmo layers exist and paint pixels while a plane is
  visible; ``show_gizmos`` off removes them; the per-plane eye hides
  just that plane's gizmo.
* **G-EXEMPT** — gizmo layers carry ``clip_exempt`` and their mappers
  hold zero clipping planes while the scene is cut.
* **G-HIT / G-MISS** — a press over the gizmo aborts and starts a
  drag; a press elsewhere does not abort (the L-PICK analog).
* **G-DRAG** — press–move–release drives the offset in the right
  direction; a LeaveEvent mid-drag ends the drag.
* **G-EVENTS** — one ``CLIP_PLANES_CHANGED`` per mutator call; the
  interactor never touches backend or panel state directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core._clip_gizmo import (
    GIZMO_LAYER_PREFIX,
    ClipGizmoRenderer,
    gizmo_geometry,
)
from apeGmsh.viewers.core._clip_gizmo_interactor import (
    install_clip_gizmo_interactor,
)
from apeGmsh.viewers.core._clip_planes import ClipPlaneSetController

_BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


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

    def GetRepeatCount(self):
        return 0

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


class _GizmoBackend:
    """Recording backend + a deterministic pixel→ray camera.

    The camera is a sheared parallel projection: every pixel maps to a
    ray origin on the ``z = 10`` plane and a fixed oblique direction —
    oblique so an X-normal plane (edge-on to a straight-down view) is
    still hittable and draggable. ``pixel_for`` inverts it, so tests
    aim at world points and never hard-code pixels.
    """

    _WPP = 0.01     # world units per pixel
    _Z0 = 10.0

    def __init__(self, viewport=(1000, 800)) -> None:
        self.viewport = viewport
        d = np.array([-0.4, 0.0, -1.0])
        self._dir = tuple(d / np.linalg.norm(d))
        self.layers: dict[str, object] = {}
        self.calls: list[tuple] = []
        self.clip_planes: tuple = ()
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
        wx = (float(x) - vw / 2.0) * self._WPP
        wy = (float(y) - vh / 2.0) * self._WPP
        return (wx, wy, self._Z0), self._dir

    def pixel_for(self, world):
        """The pixel whose ray passes through ``world``."""
        t = (float(world[2]) - self._Z0) / self._dir[2]
        ox = float(world[0]) - t * self._dir[0]
        oy = float(world[1]) - t * self._dir[1]
        vw, vh = self.viewport
        return (ox / self._WPP + vw / 2.0, oy / self._WPP + vh / 2.0)

    # -- RenderBackend subset the renderer/controller touch --
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

    def set_clip_planes(self, planes):
        self.calls.append(("set_clip_planes",))
        self.clip_planes = tuple(planes or ())

    def viewport_size(self):
        return self.viewport


class _RecordingDispatcher:
    def __init__(self) -> None:
        self.fired: list[str] = []

    def fire(self, kind: str, **_kw) -> None:
        self.fired.append(kind)


@pytest.fixture
def rig():
    """Backend + controller (one X-normal plane) + renderer + interactor."""
    backend = _GizmoBackend()
    controller = ClipPlaneSetController(backend)
    plane = controller.add((1.0, 0.0, 0.0), offset=0.0)
    renderer = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    renderer.refresh()
    interactor = install_clip_gizmo_interactor(backend, controller, renderer)
    assert interactor is not None
    return backend, controller, renderer, interactor, plane


def _geom(renderer, plane):
    return renderer.geometries()[plane.plane_id]


# =====================================================================
# G-HIT / G-MISS — the interactor claims only what it is over
# =====================================================================

def test_a_miss_does_not_abort(rig):
    """Outside a gizmo, camera and picking behave exactly as before.

    The interactor sits above navigation (11) and picking (10), so an
    over-eager abort here would break selection and orbit everywhere —
    the L-PICK guard, extended to gizmos as ADR 0083 promised.
    """
    backend, _controller, _renderer, _it, _plane = rig
    for event in ("LeftButtonPressEvent", "MouseMoveEvent",
                  "LeftButtonReleaseEvent"):
        assert backend.iren.send(event, (100, 100)) is False, event


def test_a_press_over_the_quad_aborts_and_translates(rig):
    backend, _controller, renderer, it, plane = rig
    pos = backend.pixel_for(_geom(renderer, plane).centre)
    assert it.hit(*pos) == (plane.plane_id, "translate")
    assert backend.iren.send("LeftButtonPressEvent", pos) is True


def test_a_press_over_the_tip_is_a_rotate(rig):
    backend, _controller, renderer, it, plane = rig
    pos = backend.pixel_for(_geom(renderer, plane).tip_point)
    assert it.hit(*pos) == (plane.plane_id, "rotate")
    assert backend.iren.send("LeftButtonPressEvent", pos) is True


def test_shift_lmb_over_a_gizmo_is_left_to_navigation(rig):
    """Shift+LMB is the orbit gesture; the gizmo never steals it."""
    backend, _controller, renderer, _it, plane = rig
    pos = backend.pixel_for(_geom(renderer, plane).centre)
    assert backend.iren.send("LeftButtonPressEvent", pos, shift=True) is False


def test_a_hidden_gizmo_is_not_hit(rig):
    """Eye off (or show_gizmos off) removes the hit surface with the
    pixels — a press lands on the model exactly as if no gizmo existed."""
    backend, controller, renderer, it, plane = rig
    pos = backend.pixel_for(_geom(renderer, plane).centre)

    controller.set_gizmo_visible(plane.plane_id, False)
    renderer.refresh()
    assert it.hit(*pos) == (None, None)
    assert backend.iren.send("LeftButtonPressEvent", pos) is False

    controller.set_gizmo_visible(plane.plane_id, True)
    controller.set_show_gizmos(False)
    renderer.refresh()
    assert backend.iren.send("LeftButtonPressEvent", pos) is False


def test_installed_above_navigation_and_picking(rig):
    backend, _controller, _renderer, _it, _plane = rig
    priorities = {p for _e, _cb, p in backend.iren.observers.values()}
    assert priorities == {12.0}


def test_uninstall_removes_every_observer(rig):
    backend, _controller, renderer, it, plane = rig
    pos = backend.pixel_for(_geom(renderer, plane).centre)
    it.uninstall()
    assert backend.iren.observers == {}
    assert backend.iren.send("LeftButtonPressEvent", pos) is False


# =====================================================================
# G-DRAG — press–move–release drives the offset
# =====================================================================

def test_drag_along_the_normal_moves_the_offset(rig):
    """Moving the cursor toward +X (the plane normal) increases the
    offset by the world-space distance dragged, and the release ends
    the gesture."""
    backend, controller, renderer, _it, plane = rig
    centre = np.asarray(_geom(renderer, plane).centre)
    start = backend.pixel_for(centre)
    # The pixel whose ray passes through the point one world unit
    # further along the +X normal.
    target = backend.pixel_for(centre + np.array([1.0, 0.0, 0.0]))

    backend.iren.send("LeftButtonPressEvent", start)
    assert backend.iren.send("MouseMoveEvent", target) is True
    assert controller.plane(plane.plane_id).offset == pytest.approx(1.0)

    backend.iren.send("LeftButtonReleaseEvent")
    # After the release, moves are hover only — nothing mutates.
    backend.iren.send("MouseMoveEvent", start)
    assert controller.plane(plane.plane_id).offset == pytest.approx(1.0)


def test_drag_against_the_normal_decreases_the_offset(rig):
    backend, controller, renderer, _it, plane = rig
    centre = np.asarray(_geom(renderer, plane).centre)
    backend.iren.send(
        "LeftButtonPressEvent", backend.pixel_for(centre),
    )
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(centre + np.array([-0.5, 0.0, 0.0])),
    )
    assert controller.plane(plane.plane_id).offset == pytest.approx(-0.5)


def test_leaving_the_window_ends_a_drag(rig):
    """A release delivered outside the window never reaches us; the
    0081 L2 lesson — without the LeaveEvent observer the plane follows
    the cursor forever afterwards."""
    backend, controller, renderer, _it, plane = rig
    centre = np.asarray(_geom(renderer, plane).centre)
    backend.iren.send(
        "LeftButtonPressEvent", backend.pixel_for(centre),
    )
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(centre + np.array([0.4, 0.0, 0.0])),
    )
    dragged_to = controller.plane(plane.plane_id).offset
    assert dragged_to == pytest.approx(0.4)

    backend.iren.send("LeaveEvent")
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(centre + np.array([1.5, 0.0, 0.0])),
    )
    assert controller.plane(plane.plane_id).offset == pytest.approx(dragged_to)


def test_rotate_drags_the_normal_and_pivots_about_the_quad_centre():
    """The tip handle is a virtual trackball: the normal follows the
    cursor, and ``set_pose`` re-anchors the offset so the plane pivots
    about the quad centre instead of swinging about the world origin
    (which, off-origin, would throw the plane across the model)."""
    backend = _GizmoBackend()
    controller = ClipPlaneSetController(backend)
    plane = controller.add((1.0, 0.0, 0.0), offset=0.25)
    renderer = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    renderer.refresh()
    assert install_clip_gizmo_interactor(backend, controller, renderer)

    geom = renderer.geometries()[plane.plane_id]
    centre = np.asarray(geom.centre)
    assert centre == pytest.approx([0.25, 0.0, 0.0])

    backend.iren.send(
        "LeftButtonPressEvent", backend.pixel_for(geom.tip_point),
    )
    assert backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(centre + np.array([0.0, 0.5, 0.5])),
    ) is True

    p = controller.plane(plane.plane_id)
    normal = np.asarray(p.normal)
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert normal[1] > 0.1 and normal[2] > 0.1, "normal must follow the drag"
    # The pivot: the press-time quad centre still lies on the plane.
    assert p.offset == pytest.approx(float(centre @ normal))


def test_rotating_a_flipped_plane_keeps_the_arrow_on_the_cursor():
    """The arrow points at the surviving half. Dragging the tip of a
    flipped plane must keep that side under the cursor — i.e. the
    *effective* (spec) normal follows the drag, not the stored one."""
    backend = _GizmoBackend()
    controller = ClipPlaneSetController(backend)
    plane = controller.add((1.0, 0.0, 0.0), offset=0.0)
    controller.set_flipped(plane.plane_id, True)
    renderer = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    renderer.refresh()
    assert install_clip_gizmo_interactor(backend, controller, renderer)

    geom = renderer.geometries()[plane.plane_id]
    assert np.asarray(geom.direction) == pytest.approx([-1.0, 0.0, 0.0])

    backend.iren.send(
        "LeftButtonPressEvent", backend.pixel_for(geom.tip_point),
    )
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(np.asarray(geom.centre) + np.array([0.0, 0.5, 0.0])),
    )
    p = controller.plane(plane.plane_id)
    assert p.flipped is True, "the gesture must not silently unflip"
    assert p.spec().normal[1] > 0.1, "the surviving side follows the cursor"


# =====================================================================
# G-EVENTS — one event per mutator call, and no side doors
# =====================================================================

def test_a_drag_fires_one_event_per_move_and_no_side_effects(rig):
    from apeGmsh.viewers.diagrams._dispatch import CLIP_PLANES_CHANGED

    backend, controller, renderer, _it, plane = rig
    dispatcher = _RecordingDispatcher()
    controller.dispatcher = dispatcher
    centre = np.asarray(_geom(renderer, plane).centre)

    backend.iren.send(
        "LeftButtonPressEvent", backend.pixel_for(centre),
    )
    assert dispatcher.fired == [], "a press alone mutates nothing"

    backend.calls.clear()
    for i in range(1, 21):
        backend.iren.send(
            "MouseMoveEvent",
            backend.pixel_for(centre + np.array([i * 0.05, 0.0, 0.0])),
        )
    backend.iren.send("LeftButtonReleaseEvent")

    assert dispatcher.fired == [CLIP_PLANES_CHANGED] * 20

    # The interactor's only backend traffic is the pick ray; the cut
    # itself is pushed by the controller (the owner), and the gizmo
    # layers are repainted by the renderer's own subscription — never
    # by the interactor.
    kinds = {call[0] for call in backend.calls}
    assert kinds <= {"ray", "set_clip_planes"}
    assert backend.calls.count(("set_clip_planes",)) == 20


def test_a_move_refresh_updates_actors_in_place():
    """The renderer's half of the flicker lesson (0081 L2): a drag's
    per-move refresh reuses both gizmo actors via ``update_layer`` —
    zero adds, zero removes after the first paint."""
    backend = _GizmoBackend()
    controller = ClipPlaneSetController(backend)
    plane = controller.add((1.0, 0.0, 0.0))
    renderer = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    renderer.refresh()

    backend.calls.clear()
    for i in range(1, 41):
        controller.set_offset(plane.plane_id, i * 0.01)
        renderer.refresh()
    kinds = [c[0] for c in backend.calls]
    assert "add" not in kinds and "remove" not in kinds
    assert kinds.count("update") == 80        # quad + arrow, 40 moves


# =====================================================================
# G-RENDER — gizmos in the frame (offscreen pixels)
# =====================================================================

_PAINTED = 12


@pytest.fixture
def gizmo_backend():
    """Offscreen backend on a small black canvas, for pixel counting."""
    import pyvista as pv

    from apeGmsh.viewers.backends import PyVistaQtBackend
    try:
        plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    except Exception:                                   # pragma: no cover
        pytest.skip("no offscreen render context")
    plotter.background_color = "black"
    yield PyVistaQtBackend(plotter)
    plotter.close()


def _painted(backend) -> int:
    """Painted-pixel count of the current frame (camera pinned)."""
    camera = backend.plotter.camera
    camera.parallel_projection = True
    camera.position = (0.0, 0.0, 10.0)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = (0.0, 1.0, 0.0)
    camera.parallel_scale = 3.0
    camera.clipping_range = (0.1, 100.0)
    backend.plotter.render()
    img = np.asarray(
        backend.plotter.screenshot(
            return_img=True, transparent_background=False,
        ),
    )
    return int((img.astype(np.int32).sum(axis=2) > _PAINTED).sum())


def _gizmo_layer_ids(backend) -> set:
    return {
        layer_id
        for layer_id in backend._handles
        if layer_id.startswith(GIZMO_LAYER_PREFIX)
    }


def test_gizmo_renders_and_the_toggles_remove_it(
    gizmo_backend, requires_gl_actor_clear,
):
    """The whole G-RENDER story on one plane: pixels appear with the
    gizmo, vanish with ``show_gizmos`` off, vanish with the eye off."""
    controller = ClipPlaneSetController(gizmo_backend)
    renderer = ClipGizmoRenderer(gizmo_backend, controller, bbox=_BBOX)
    # +Z normal: the quad faces the pinned camera head-on.
    plane = controller.add((0.0, 0.0, 1.0), offset=0.0)
    renderer.refresh()

    assert _gizmo_layer_ids(gizmo_backend) == {
        f"{GIZMO_LAYER_PREFIX}{plane.plane_id}:quad",
        f"{GIZMO_LAYER_PREFIX}{plane.plane_id}:arrow",
    }
    with_gizmo = _painted(gizmo_backend)
    assert with_gizmo > 0, "the gizmo must actually paint pixels"

    controller.set_show_gizmos(False)
    renderer.refresh()
    assert _gizmo_layer_ids(gizmo_backend) == set()
    assert _painted(gizmo_backend) == 0

    controller.set_show_gizmos(True)
    renderer.refresh()
    assert _painted(gizmo_backend) == with_gizmo

    controller.set_gizmo_visible(plane.plane_id, False)
    renderer.refresh()
    assert _gizmo_layer_ids(gizmo_backend) == set()
    assert _painted(gizmo_backend) == 0


def test_the_eye_hides_only_its_own_plane(gizmo_backend):
    controller = ClipPlaneSetController(gizmo_backend)
    renderer = ClipGizmoRenderer(gizmo_backend, controller, bbox=_BBOX)
    first = controller.add((0.0, 0.0, 1.0))
    second = controller.add((1.0, 0.0, 0.0))
    renderer.refresh()
    assert len(_gizmo_layer_ids(gizmo_backend)) == 4

    controller.set_gizmo_visible(first.plane_id, False)
    renderer.refresh()
    remaining = _gizmo_layer_ids(gizmo_backend)
    assert remaining == {
        f"{GIZMO_LAYER_PREFIX}{second.plane_id}:quad",
        f"{GIZMO_LAYER_PREFIX}{second.plane_id}:arrow",
    }


def test_a_deleted_plane_takes_its_gizmo_along(
    gizmo_backend, requires_gl_actor_clear,
):
    controller = ClipPlaneSetController(gizmo_backend)
    renderer = ClipGizmoRenderer(gizmo_backend, controller, bbox=_BBOX)
    plane = controller.add((0.0, 0.0, 1.0))
    renderer.refresh()
    assert _painted(gizmo_backend) > 0

    controller.remove(plane.plane_id)
    renderer.refresh()
    assert _gizmo_layer_ids(gizmo_backend) == set()
    assert _painted(gizmo_backend) == 0


# =====================================================================
# G-EXEMPT — the cut never slices its own UI
# =====================================================================

def _n_planes(handle) -> int:
    return int(handle.actor.GetMapper().GetNumberOfClippingPlanes())


def _brick_layer():
    from apeGmsh.viewers.scene_ir import CellBlocks, MeshLayer, PointSet

    pts = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=float,
    )
    return MeshLayer(
        layer_id="brick",
        points=PointSet(pts),
        cells=CellBlocks({"hexahedron": np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}),
    )


def test_gizmo_layers_are_exempt_from_the_cut(gizmo_backend):
    """The regression pin for the exemption: while the scene is cut,
    ordinary mappers carry the planes and gizmo mappers carry none —
    at creation (stamping) and on every re-cut alike."""
    brick = gizmo_backend.add_layer(_brick_layer())
    controller = ClipPlaneSetController(gizmo_backend)
    renderer = ClipGizmoRenderer(gizmo_backend, controller, bbox=_BBOX)

    plane = controller.add((1.0, 0.0, 0.0))     # the cut is live now
    renderer.refresh()                          # gizmos created under it
    quad, arrow = renderer._handles[plane.plane_id]

    assert quad.clip_exempt is True and arrow.clip_exempt is True
    assert _n_planes(brick) == 1, "the model is cut"
    assert _n_planes(quad) == 0, "the gizmo quad must never be cut"
    assert _n_planes(arrow) == 0, "the gizmo arrow must never be cut"

    # A later re-cut (any mutator) must keep the exemption.
    controller.set_offset(plane.plane_id, 0.4)
    renderer.refresh()
    assert _n_planes(brick) == 1
    assert _n_planes(quad) == 0 and _n_planes(arrow) == 0

    # A second plane cuts the model twice — and the first plane's
    # gizmo still not at all (a gizmo sliced by ANOTHER plane is as
    # useless as one sliced by its own).
    controller.add((0.0, 0.0, 1.0))
    assert _n_planes(brick) == 2
    assert _n_planes(quad) == 0 and _n_planes(arrow) == 0


def test_gizmo_layers_declare_the_exemption_on_the_ir():
    """``clip_exempt`` is carried by the scene-IR layers themselves —
    the backend honours a declaration, it does not guess by name."""
    from apeGmsh.viewers.core._clip_gizmo import _arrow_layer, _quad_layer
    from apeGmsh.viewers.scene_ir import MeshLayer

    class _Plane:
        plane_id = "plane-1"
        normal = (0.0, 0.0, 1.0)
        offset = 0.0
        flipped = False

    geom = gizmo_geometry(_Plane(), _BBOX)
    assert _quad_layer(geom).clip_exempt is True
    assert _arrow_layer(geom).clip_exempt is True
    assert _quad_layer(geom).pickable is False
    assert _arrow_layer(geom).pickable is False
    # And the default stays False — ordinary layers keep getting cut.
    assert MeshLayer.__dataclass_fields__["clip_exempt"].default is False


# =====================================================================
# Geometry sanity — the quad tracks the plane it projects
# =====================================================================

def test_quad_centre_sits_on_the_plane_and_spans_the_bbox():
    class _Plane:
        plane_id = "plane-1"
        normal = (1.0, 0.0, 0.0)
        offset = 0.6
        flipped = False

    geom = gizmo_geometry(_Plane(), _BBOX)
    centre = np.asarray(geom.centre)
    assert float(centre @ np.asarray(geom.normal)) == pytest.approx(0.6)
    # The quad covers the bbox cross-section (±1 here) with margin.
    assert geom.half_u >= 1.0 and geom.half_v >= 1.0
    # The arrow points into the surviving half; flip reverses it.
    assert np.asarray(geom.direction) == pytest.approx([1.0, 0.0, 0.0])

    class _Flipped(_Plane):
        flipped = True

    assert np.asarray(
        gizmo_geometry(_Flipped(), _BBOX).direction,
    ) == pytest.approx([-1.0, 0.0, 0.0])


# =====================================================================
# S2 adversarial review — regressions C1 (drag gain) and C2 (cursor)
# =====================================================================

def test_translate_gain_is_bounded_near_face_on_viewing():
    """C1: ``1 - (n.d)^2`` divides the drag projection, so a camera
    swinging onto the plane normal — which the toolbar's axis view
    snaps do in one click — used to make the gesture explode. Measured
    before the floor: 2.1 units per pixel at 1 degree off face-on, on a
    10-unit model; 21 units at 0.1 degrees. The floor must bound it."""
    import math

    from apeGmsh.viewers.core._clip_gizmo import (
        DRAG_MIN_SEPARATION,
        axis_param,
    )

    centre = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    pixel = 0.0375          # world shift of the ray for a 1-px move

    def delta_per_pixel(deg: float) -> float:
        a = math.radians(deg)
        eye = np.array([math.sin(a), 0.0, math.cos(a)]) * 30.0
        d = -eye / np.linalg.norm(eye)
        right = np.array([math.cos(a), 0.0, -math.sin(a)])
        s0 = axis_param(
            centre, normal, eye, d, min_separation=DRAG_MIN_SEPARATION,
        )
        s1 = axis_param(
            centre, normal, eye + right * pixel, d,
            min_separation=DRAG_MIN_SEPARATION,
        )
        return abs(s1 - s0)

    # Bounded everywhere, including angles that used to fling the plane
    # clean off a 10-unit model.
    for deg in (90.0, 30.0, 5.0, 1.0, 0.1, 0.0):
        assert delta_per_pixel(deg) < 0.5, f"{deg} deg is hypersensitive"

    # Away from the cone the projection is untouched — the floor must
    # not tax the ordinary case.
    plain = axis_param(
        centre, normal, np.array([30.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    )
    floored = axis_param(
        centre, normal, np.array([30.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]), min_separation=DRAG_MIN_SEPARATION,
    )
    assert plain == pytest.approx(floored)

    # The hit-test path keeps the exact projection and its None contract.
    assert axis_param(
        centre, normal, np.array([0.0, 0.0, 30.0]),
        np.array([0.0, 0.0, -1.0]),
    ) is None


def test_a_face_on_drag_does_not_fling_the_plane():
    """C1 end to end: the same guard, but through the real interactor —
    this is the one that fails if the drag path stops passing the
    floor. A Z-normal plane viewed ~1 degree off its own normal (one
    click of the toolbar's Top view snap, plus a nudge) must not leave
    the model on a one-pixel move."""
    import math

    tilt = math.sin(math.radians(1.0))

    class _FaceOnBackend(_GizmoBackend):
        def __init__(self) -> None:
            super().__init__()
            d = np.array([tilt, 0.0, -1.0])
            self._dir = tuple(d / np.linalg.norm(d))

    backend = _FaceOnBackend()
    controller = ClipPlaneSetController(backend)
    plane = controller.add((0.0, 0.0, 1.0), offset=0.0)
    renderer = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    renderer.refresh()
    interactor = install_clip_gizmo_interactor(backend, controller, renderer)
    assert interactor is not None

    geom = renderer.geometries()[plane.plane_id]
    # Out on the quad, clear of the tip: viewed face-on the arrow points
    # straight at the camera, so a press at the CENTRE grabs the rotate
    # handle and says nothing about translate sensitivity.
    grab = np.asarray(geom.centre) + 0.8 * geom.half_u * np.asarray(geom.u)
    start = backend.pixel_for(grab)
    assert interactor.hit(*start) == (plane.plane_id, "translate"), (
        "this test is only meaningful while the press grabs translate"
    )

    assert backend.iren.send("LeftButtonPressEvent", start) is True, (
        "a press on the gizmo must be claimed even face-on — bailing "
        "here hands the click to the picker"
    )
    backend.iren.send("MouseMoveEvent", (start[0] + 1.0, start[1]))
    moved = abs(controller.plane(plane.plane_id).offset)
    # The model spans 2 units; unfloored, one pixel moved it 0.57.
    assert moved < 0.1, f"one pixel moved the plane {moved:.3f} units"


def test_a_gizmo_hover_does_not_erase_the_legend_cursor():
    """C2: both hover interactors run on every move and neither aborts
    on a miss, so the one that runs last used to win the window cursor.
    Sliding off a gizmo onto a legend left the legend showing the
    default arrow until the pointer left and came back."""
    import vtk

    from apeGmsh.viewers.core._cursor_arbiter import request_cursor

    class _Window:
        def __init__(self) -> None:
            self.cursor = vtk.VTK_CURSOR_DEFAULT

        def SetCurrentCursor(self, value):    # noqa: N802 (VTK spelling)
            self.cursor = value

    window = _Window()

    # Hover a gizmo, then slide straight onto a legend: the legend asks
    # first (it is installed first), the gizmo withdraws second.
    request_cursor(window, "clip_gizmo", vtk.VTK_CURSOR_SIZEALL)
    assert window.cursor == vtk.VTK_CURSOR_SIZEALL
    request_cursor(window, "legend", vtk.VTK_CURSOR_SIZENE)
    request_cursor(window, "clip_gizmo", None)
    assert window.cursor == vtk.VTK_CURSOR_SIZENE, "gizmo erased the legend"

    # Leaving both restores the default.
    request_cursor(window, "legend", None)
    assert window.cursor == vtk.VTK_CURSOR_DEFAULT

    # A gizmo request never outranks a live legend request, whichever
    # order they arrive in (the legend wins the press, so it wins here).
    request_cursor(window, "clip_gizmo", vtk.VTK_CURSOR_HAND)
    request_cursor(window, "legend", vtk.VTK_CURSOR_SIZEALL)
    assert window.cursor == vtk.VTK_CURSOR_SIZEALL


def test_both_hover_interactors_route_through_the_arbiter():
    """C2, end to end: the guard above is only worth anything while
    both interactors actually go through the arbiter — a direct
    ``SetCurrentCursor`` in either one restores the stomp."""
    import vtk

    from apeGmsh.viewers.core._clip_gizmo_interactor import ClipGizmoInteractor
    from apeGmsh.viewers.core._legend_interactor import LegendInteractor

    backend = _GizmoBackend()
    window = backend.render_window

    gizmo = ClipGizmoInteractor(backend, None, None)
    legend = LegendInteractor.__new__(LegendInteractor)
    legend._cursor = None
    legend._backend = backend

    # Hover the gizmo, then slide onto a legend in one move: the legend
    # asks (it runs first), then the gizmo withdraws (it runs second).
    gizmo._set_cursor("translate")
    assert window.cursor == vtk.VTK_CURSOR_SIZEALL
    legend._set_cursor("move")
    gizmo._set_cursor(None)
    assert window.cursor == vtk.VTK_CURSOR_SIZEALL, (
        "the gizmo's miss erased the legend's hover cursor"
    )
