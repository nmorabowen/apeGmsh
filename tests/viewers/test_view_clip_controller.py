"""ADR 0098 R1 slice 1 — the session serves the ADR 0083 clip contract.

The gizmo renderer and its interactor are working, tested code
(``test_clip_gizmo.py``, ``test_clip_gizmo_interactor.py``); what they
have never had on the session path is a **controller** to read and
write. :class:`ViewClipController` is that adapter, and these tests
drive it through the REAL renderer and the REAL interactor rather than
calling its five methods directly — the house rule is to assert on the
affordance, not on the setter. A test that only checked that
``set_offset`` wrote ``offset`` would pass against an adapter the gizmo
could not actually use.

Everything here is headless: the renderer needs a backend with
``add_layer`` / ``update_layer`` / ``remove_layer``, and the
interactor's gesture handlers need ``display_to_world_ray`` plus a
caller object. No window, no VTK render, no Qt.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results.session._views import MeshView
from apeGmsh.viewers.core._clip_gizmo import MODE_TRANSLATE, ClipGizmoRenderer
from apeGmsh.viewers.core._clip_gizmo_interactor import ClipGizmoInteractor
from apeGmsh.viewers.session._clip_control import ViewClipController

BBOX = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)


class FakeBackend:
    """Records layers; unprojects any pixel to a ray it is handed."""

    def __init__(self) -> None:
        self.layers: "dict[int, object]" = {}
        self._next = 0
        self.ray: tuple = ((5.0, 5.0, 50.0), (0.0, 0.0, -1.0))
        self.updates = 0

    def add_layer(self, layer):
        self._next += 1
        self.layers[self._next] = layer
        return self._next

    def update_layer(self, handle, layer):
        self.layers[handle] = layer
        self.updates += 1

    def remove_layer(self, handle):
        self.layers.pop(handle, None)

    def display_to_world_ray(self, x, y):
        return self.ray


class FakeCaller:
    """The VTK interactor object the gesture handlers are given."""

    def __init__(self, pos=(0, 0), shift=False) -> None:
        self.pos = pos
        self.shift = shift

    def GetShiftKey(self):
        return self.shift

    def GetEventPosition(self):
        return self.pos

    def GetCommand(self, _tag):
        return None


def _view(**clip):
    view = MeshView(pane_id="mesh-1")
    # Normal +X, viewed down -Z. Both choices are load-bearing: an
    # arrow along the VIEW axis puts the rotate tip on the press ray
    # (so every press grabs ROTATE), and a translate along the view
    # axis is the degenerate face-on projection ``DRAG_MIN_SEPARATION``
    # exists to bound. Crosswise is the ordinary case.
    view.add_clip((1.0, 0.0, 0.0), offset=5.0, **clip)
    return view


def _wired(view):
    """A renderer + interactor pair over one view, refreshed once."""
    backend = FakeBackend()
    controller = ViewClipController(view)
    gizmos = ClipGizmoRenderer(backend, controller, bbox=BBOX)
    gizmos.refresh()
    interactor = ClipGizmoInteractor(backend, controller, gizmos)
    return backend, controller, gizmos, interactor


# ---------------------------------------------------------------------
# The renderer reads the view
# ---------------------------------------------------------------------

def test_renderer_draws_a_gizmo_for_a_view_clip():
    """The affordance: a clip on a session view produces gizmo actors.

    This is A6.6's R1 defect in one assertion — before the adapter
    there was no controller, so ``refresh`` drew nothing and an
    interactor had nothing to grab.
    """
    backend, _c, gizmos, _i = _wired(_view())
    assert gizmos.geometries(), "no gizmo geometry for a view with a clip"
    assert len(backend.layers) == 2, "expected a quad layer and an arrow layer"


def test_gizmo_geometry_tracks_the_view_record():
    """Move the plane through the VIEW and the gizmo follows."""
    view = _view()
    _b, _c, gizmos, _i = _wired(view)
    before = gizmos.geometries()["clip-1"].centre

    view.set_clip("clip-1", offset=8.0)
    gizmos.refresh()
    after = gizmos.geometries()["clip-1"].centre

    assert before[0] == pytest.approx(5.0)
    assert after[0] == pytest.approx(8.0)


def test_per_plane_gizmo_visible_hides_only_that_gizmo():
    view = _view()
    view.add_clip((0.0, 0.0, 1.0), offset=5.0, gizmo_visible=False)
    _b, _c, gizmos, _i = _wired(view)
    assert set(gizmos.geometries()) == {"clip-1"}


def test_show_gizmos_is_derived_and_hiding_every_plane_clears_the_set():
    """A5.3's rule: placement is state, EXISTENCE stays derived.

    ``show_gizmos`` has no field of its own, so the master toggle is
    "clear every plane's flag" — and that must actually empty the drawn
    set, which is what makes the derivation equivalent to the old
    viewer-wide flag rather than merely cheaper.
    """
    view = _view()
    _b, controller, gizmos, _i = _wired(view)
    assert controller.show_gizmos is True

    view.set_clip("clip-1", gizmo_visible=False)
    assert controller.show_gizmos is False
    gizmos.refresh()
    assert gizmos.geometries() == {}


def test_a_view_with_no_clips_draws_nothing_and_does_not_raise():
    backend = FakeBackend()
    controller = ViewClipController(MeshView(pane_id="mesh-1"))
    gizmos = ClipGizmoRenderer(backend, controller, bbox=BBOX)
    gizmos.refresh()
    assert controller.show_gizmos is False
    assert gizmos.geometries() == {}
    assert backend.layers == {}


# ---------------------------------------------------------------------
# The interactor writes the view
# ---------------------------------------------------------------------

def test_hit_test_finds_the_quad():
    """The adapter feeds a hit surface the interactor can actually use."""
    _b, _c, _g, interactor = _wired(_view())
    assert interactor.hit(0, 0) == ("clip-1", MODE_TRANSLATE)


def test_translate_drag_writes_the_offset_onto_the_VIEW():
    """A synthesized press-move changes the SESSION record, not just a
    controller — the A5.6-2 shape of the criterion.

    The plane is x = 5 with normal +X, and the ray runs down -Z, so
    the two are perpendicular (``axis_param``'s ``b = 0``) and the
    projection onto the normal axis is just the ray's x. Sliding the
    ray from x = 5 to x = 7 therefore moves the plane by exactly +2.
    """
    view = _view()
    backend, _c, _g, interactor = _wired(view)

    caller = FakeCaller(pos=(0, 0))
    interactor._on_lmb_press(caller, None)
    assert interactor.is_dragging, "press over the quad did not start a drag"

    backend.ray = ((7.0, 5.0, 50.0), (0.0, 0.0, -1.0))
    interactor._on_mouse_move(caller, None)

    assert view.clips[0].offset == pytest.approx(7.0)
    assert view.clips[0].plane_id == "clip-1", "identity must not move"


def test_rotate_drag_writes_normal_and_offset_in_ONE_tick():
    """ADR 0083 S2's transient-plane rule survives the port.

    Two writes would publish a plane with the new normal and the stale
    offset for one frame. The view's change tick is countable, so this
    is measured rather than assumed.
    """
    view = _view()
    backend, _c, gizmos, interactor = _wired(view)

    geom = gizmos.geometries()["clip-1"]
    tip = np.asarray(geom.tip_point)
    # Fire straight down -Z through the tip so the press grabs ROTATE.
    backend.ray = ((float(tip[0]), float(tip[1]), 50.0), (0.0, 0.0, -1.0))
    caller = FakeCaller(pos=(0, 0))
    interactor._on_lmb_press(caller, None)
    assert interactor.is_dragging, "press over the tip did not start a rotate"

    ticks: "list[int]" = []
    view._notify = lambda: ticks.append(1)
    # Swing the ray off-axis so the trackball point moves.
    backend.ray = (
        (float(geom.centre[0]) + 1.0, float(geom.centre[1]) + 3.0, 50.0),
        (0.0, 0.0, -1.0),
    )
    interactor._on_mouse_move(caller, None)

    assert len(ticks) == 1, (
        "a rotate move must be ONE mutation — two ticks means a "
        "transient plane (new normal, stale offset) was published"
    )
    assert view.clips[0].normal != (1.0, 0.0, 0.0)


def test_a_drag_on_a_plane_deleted_mid_gesture_is_a_no_op():
    """The one race a live gesture can legitimately lose.

    ``MeshView.set_clip`` raises ``KeyError`` for an unknown plane, on
    purpose — a scripted edit naming a missing plane is a bug. But
    these mutators run inside a VTK observer callback, so the adapter
    must absorb exactly that one failure rather than let it escape.
    """
    view = _view()
    backend, controller, _g, interactor = _wired(view)
    interactor._on_lmb_press(FakeCaller(pos=(0, 0)), None)

    view.remove_clip("clip-1")
    backend.ray = ((7.0, 5.0, 50.0), (0.0, 0.0, -1.0))
    interactor._on_mouse_move(FakeCaller(pos=(0, 0)), None)   # must not raise

    assert view.clips == ()
    assert controller.plane("clip-1") is None


def test_the_swallow_is_scoped_to_KeyError_and_nothing_else():
    """A broad ``except Exception`` here would be invisible.

    The adapter absorbs ``KeyError`` because a plane can vanish under
    an in-flight gesture. Widening that to ``Exception`` would silently
    eat every real bug — a malformed normal, a frozen-record violation
    — inside a VTK callback, where there is no traceback to find it by.
    A two-component normal reaches ``_unit`` and raises ``IndexError``;
    that must still escape.
    """
    controller = ViewClipController(_view())
    with pytest.raises(IndexError):
        controller.set_pose("clip-1", (1.0, 0.0), 5.0)


def test_a_scripted_edit_naming_a_missing_plane_still_raises():
    """The swallow is scoped to the gesture path, not to the view.

    Without this, "the adapter absorbs KeyError" could be read as
    licence to make ``set_clip`` quiet everywhere, which would hide the
    scripted-edit bug the loud version exists to catch.
    """
    view = _view()
    with pytest.raises(KeyError):
        view.set_clip("nope", offset=1.0)


def test_shift_press_is_left_to_navigation():
    _b, _c, _g, interactor = _wired(_view())
    interactor._on_lmb_press(FakeCaller(pos=(0, 0), shift=True), None)
    assert not interactor.is_dragging


def test_a_miss_starts_no_drag():
    backend, _c, _g, interactor = _wired(_view())
    backend.ray = ((500.0, 500.0, 50.0), (0.0, 0.0, -1.0))
    interactor._on_lmb_press(FakeCaller(pos=(0, 0)), None)
    assert not interactor.is_dragging


# ---------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------

def test_the_contract_is_exactly_the_five_members_both_consumers_use():
    controller = ViewClipController(_view())
    for name in ("plane", "planes", "show_gizmos", "set_offset", "set_pose"):
        assert hasattr(controller, name), f"missing contract member {name!r}"


def test_planes_hands_out_the_view_records_themselves():
    """No copy, no shadow: a second plane record is a second thing to
    keep in sync, which is what ADR 0098 §3 moved ownership to avoid."""
    view = _view()
    controller = ViewClipController(view)
    assert controller.planes() == list(view.clips)
    assert controller.plane("clip-1") is view.clips[0]
    assert controller.plane("nope") is None
