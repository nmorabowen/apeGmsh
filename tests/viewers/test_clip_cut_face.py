"""Contour on the cut face — the interior in the picture (ADR 0083 S3).

The ADR's *F1 revisited* evidence is what this file exists to pin: a
section plane through a solid used to open into an **empty box**,
because render-time clipping discards fragments and a contour renders
only the mesh's exterior skin. Same discipline as its two siblings
(assert the picture, not the call):

* **F-FILLED** — the headline. A solid whose field is zero on every
  exterior face and nonzero inside: uncut it shows nothing, cut it
  shows nothing *until* the cap is on, and then the interior field is
  actually there in pixels. The middle frame is the pre-S3 picture,
  measured rather than remembered.
* **F-EXEMPT** — the cap carries ``clip_exempt`` and its mapper holds
  zero clipping planes while the scene is cut (it is coplanar with its
  own plane; mapper-clipping it is a coin toss). A second plane still
  trims it — geometrically.
* **F-LUT** — the cap's ``LutSpec`` *is* the contour's. A cut face on a
  different scale would be a lie about the same quantity.
* **F-SETTLED** — no slicing during a gizmo drag, exactly one pass per
  source on release; a step change recomputes.
* **F-TOGGLE** — the panel checkbox turns it on and off, and off leaves
  the cut itself intact.
* **F-SESSION** — the flag round-trips; a v9 session loads with the
  default.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core._clip_cut_face import (
    CUT_FACE_LAYER_PREFIX,
    ClipCutFaceRenderer,
)
from apeGmsh.viewers.core._clip_gizmo import ClipGizmoRenderer
from apeGmsh.viewers.core._clip_gizmo_interactor import (
    install_clip_gizmo_interactor,
)
from apeGmsh.viewers.core._clip_planes import ClipPlaneSetController
from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ClipPlaneSpec,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarField,
)

_BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)

#: Anything brighter than this on a black background is painted.
_PAINTED = 12


# =====================================================================
# A solid whose field lives entirely in its interior
# =====================================================================

def _solid_layer(n: int = 7, *, layer_id: str = "solid") -> MeshLayer:
    """A hex box on ``[-1, 1]^3`` carrying a bubble field.

    ``cos(pi x / 2) cos(pi y / 2) cos(pi z / 2)``: exactly zero on every
    one of the six faces, 1.0 at the centre. The exact shape the ADR
    measured the empty box on — from outside there is nothing to see,
    which is the whole reason section planes exist.
    """
    xs = np.linspace(-1.0, 1.0, n)
    gx, gy, gz = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def idx(i, j, k):
        return (i * n + j) * n + k

    hexes = np.asarray([
        [
            idx(i, j, k), idx(i + 1, j, k),
            idx(i + 1, j + 1, k), idx(i, j + 1, k),
            idx(i, j, k + 1), idx(i + 1, j, k + 1),
            idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
        ]
        for i in range(n - 1) for j in range(n - 1) for k in range(n - 1)
    ])
    bubble = (
        np.cos(np.pi * pts[:, 0] / 2.0)
        * np.cos(np.pi * pts[:, 1] / 2.0)
        * np.cos(np.pi * pts[:, 2] / 2.0)
    )
    return MeshLayer(
        layer_id=layer_id,
        points=PointSet(pts),
        cells=CellBlocks({"hexahedron": hexes}),
        fields=(ScalarField("v", bubble, "point"),),
        color=ColorSpec(
            mode="by_array", array_name="v",
            lut=LutSpec(name="jet", vmin=0.0, vmax=1.0),
        ),
    )


@pytest.fixture
def cut_face_backend():
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


def _frame(backend) -> np.ndarray:
    """Render looking straight down ``-z`` and return the RGB frame.

    Pinned, not reset: clipping changes the scene bounds, and a camera
    that refits would reframe between the frames a test compares.
    """
    camera = backend.plotter.camera
    camera.parallel_projection = True
    camera.position = (0.0, 0.0, 10.0)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = (0.0, 1.0, 0.0)
    camera.parallel_scale = 1.4
    camera.clipping_range = (0.1, 100.0)
    backend.plotter.render()
    return np.asarray(
        backend.plotter.screenshot(
            return_img=True, transparent_background=False,
        ),
    )


def _painted(img: np.ndarray) -> int:
    return int((img.astype(np.int32).sum(axis=2) > _PAINTED).sum())


def _hot(img: np.ndarray) -> int:
    """Pixels the warm end of ``jet`` painted — i.e. a nonzero field.

    ``jet`` maps 0 to dark blue and the top of the range to red, so
    "red channel clearly above blue" is exactly "this pixel is showing
    interior values". Zero-valued geometry and the black background
    both score zero here, which is what makes this the F-FILLED
    assertion rather than a paint-count.
    """
    rgb = img.astype(np.int32)
    return int(((rgb[..., 0] - rgb[..., 2]) > 40).sum())


def _n_planes(handle) -> int:
    return int(handle.actor.GetMapper().GetNumberOfClippingPlanes())


def _sources_for(key: str, handle, color):
    return lambda: [(key, handle, color)]


# =====================================================================
# F-FILLED — the interior is actually on screen
# =====================================================================

def test_the_cut_face_shows_the_interior_field(cut_face_backend):
    """The whole S3 story in three frames.

    Uncut: a blank blue block (the field is zero on every face).
    Cut, no cap: still blank — an empty box, which is the defect the
    ADR promoted this slice for. Cap on: the bubble is right there.
    """
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)

    uncut = _frame(cut_face_backend)
    assert _painted(uncut) > 0, "the block must render at all"
    assert _hot(uncut) == 0, "a surface contour of this field shows nothing"

    # Keep the z < 0 half; the cut face at z = 0 faces the camera.
    controller = ClipPlaneSetController(cut_face_backend)
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_only = _frame(cut_face_backend)
    assert _painted(cut_only) > 0, "the kept half must still render"
    assert _hot(cut_only) == 0, (
        "pre-S3 picture: cutting a solid opens an EMPTY BOX — if this "
        "ever paints hot pixels the test has stopped measuring S3"
    )

    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    controller.set_cut_face(True)
    cut_faces.refresh()

    assert cut_faces.layer_ids() == {
        f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour",
    }
    capped = _frame(cut_face_backend)
    hot = _hot(capped)
    assert hot > 200, (
        f"the cut face must show the interior field; only {hot} hot pixels"
    )


def test_removing_the_plane_takes_its_cut_face_along(cut_face_backend):
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    cut_faces.refresh()
    assert _hot(_frame(cut_face_backend)) > 200

    controller.remove(plane.plane_id)
    cut_faces.refresh()
    assert cut_faces.layer_ids() == set()
    assert _hot(_frame(cut_face_backend)) == 0

    # Deactivating (rather than deleting) does the same.
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces.refresh()
    assert _hot(_frame(cut_face_backend)) > 200
    controller.set_active(plane.plane_id, False)
    cut_faces.refresh()
    assert cut_faces.layer_ids() == set()


def test_a_plane_that_misses_the_model_grows_no_cut_face(cut_face_backend):
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    controller.add((0.0, 0.0, -1.0), offset=50.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    cut_faces.refresh()
    assert cut_faces.layer_ids() == set()


# =====================================================================
# F-EXEMPT — the cap is not sliced by the plane it caps
# =====================================================================

def test_the_cut_face_is_exempt_from_its_own_plane(cut_face_backend):
    """A cap is coplanar with its plane, so whether its fragments pass
    that plane's mapper test is a floating-point coin toss — it would
    flicker or vanish outright. The exemption is the fix, and it is
    declared on the IR, not guessed from a layer name."""
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    cut_faces.refresh()

    cap_id = f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"
    cap = cut_faces._handles[cap_id]
    assert cap.clip_exempt is True
    assert _n_planes(handle) == 1, "the model is cut"
    assert _n_planes(cap) == 0, "the cut face must never be cut"

    # A later re-cut keeps the exemption, and a second plane cuts the
    # model twice while still not touching the cap's mapper.
    controller.set_offset(plane.plane_id, 0.2)
    cut_faces.refresh()
    assert _n_planes(cut_faces._handles[cap_id]) == 0
    controller.add((1.0, 0.0, 0.0), offset=0.0)
    cut_faces.refresh()
    assert _n_planes(handle) == 2
    assert _n_planes(cut_faces._handles[cap_id]) == 0


def test_a_second_plane_trims_the_cut_face_geometrically(cut_face_backend):
    """The corner cut: exempt from *mapper* clipping is not exempt from
    the other planes. They trim the polygon in the slice instead, where
    coplanarity is not at issue."""
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    cut_faces.refresh()
    cap_id = f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"
    full = cut_faces._handles[cap_id].dataset
    assert float(np.asarray(full.points)[:, 0].min()) < -0.5

    # Keep only x >= 0: the cap must stop at the second plane.
    controller.add((1.0, 0.0, 0.0), offset=0.0)
    cut_faces.refresh()
    trimmed = np.asarray(cut_faces._handles[cap_id].dataset.points)
    assert float(trimmed[:, 0].min()) == pytest.approx(0.0, abs=1e-6)
    assert float(trimmed[:, 0].max()) == pytest.approx(1.0, abs=1e-6)


def test_the_cut_face_steps_clear_of_the_gizmo_quad(cut_face_backend):
    """Both the cap and the S2 gizmo quad live on the plane, and two
    coplanar surfaces fight fragment by fragment — measured live, the
    cap came back speckled with patches of translucent quad. The cap
    steps a hair into the *discarded* half, where the clip has left
    nothing else to draw, so it wins deterministically."""
    from apeGmsh.viewers.core._clip_cut_face import _CAP_NUDGE_FRAC

    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    cut_faces.refresh()

    cap = cut_faces._handles[
        f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"
    ]
    z = np.asarray(cap.dataset.points)[:, 2]
    # Surviving normal is -z, so "discarded" is +z. The step is real…
    assert float(z.min()) > 0.0
    # …and small enough to read as "on the plane".
    assert float(z.max()) < 4.0 * _CAP_NUDGE_FRAC * 3.0
    # The gizmo quad, meanwhile, stays exactly on the plane.
    gizmos = ClipGizmoRenderer(cut_face_backend, controller, bbox=_BBOX)
    gizmos.refresh()
    assert gizmos.geometries()[plane.plane_id].centre[2] == pytest.approx(0.0)


# =====================================================================
# F-LUT — one quantity, one scale
# =====================================================================

def test_the_cut_face_shares_the_contours_colour_mapping(cut_face_backend):
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    plane = controller.add((0.0, 0.0, -1.0), offset=0.0)

    emitted: dict = {}
    original_add = cut_face_backend.add_layer

    def _record(scene_layer):
        emitted[scene_layer.layer_id] = scene_layer
        return original_add(scene_layer)

    cut_face_backend.add_layer = _record            # type: ignore[assignment]
    ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    ).refresh()

    cap = emitted[f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"]
    assert cap.color.mode == "by_array"
    assert cap.color.array_name == layer.color.array_name
    # The LutSpec itself, not merely "some colour" — a cap on a
    # different clim or cmap is a lie about the same quantity.
    assert cap.color.lut == layer.color.lut
    assert cap.field_named(layer.color.array_name) is not None
    assert cap.pickable is False, "the cap must not eat model picks"


# =====================================================================
# F-SETTLED — sliced on release and on step change, never per tick
# =====================================================================

class _Handle:
    def __init__(self, layer_id: str) -> None:
        self.layer_id = layer_id
        self.visible = True


class _CountingBackend:
    """Recording backend + the deterministic pixel→ray camera from
    ``test_clip_gizmo`` (a sheared parallel projection), plus a
    ``slice_layer`` that counts instead of slicing."""

    _WPP = 0.01
    _Z0 = 10.0

    def __init__(self) -> None:
        self.viewport = (1000, 800)
        d = np.array([-0.4, 0.0, -1.0])
        self._dir = tuple(d / np.linalg.norm(d))
        self.layers: dict = {}
        self.slices = 0
        self.iren = _Iren()
        outer = self
        self.plotter = type(
            "P", (), {"iren": type("W", (), {"interactor": outer.iren})()},
        )()

    def display_to_world_ray(self, x, y):
        vw, vh = self.viewport
        return (
            ((float(x) - vw / 2.0) * self._WPP,
             (float(y) - vh / 2.0) * self._WPP, self._Z0),
            self._dir,
        )

    def pixel_for(self, world):
        t = (float(world[2]) - self._Z0) / self._dir[2]
        vw, vh = self.viewport
        return (
            (float(world[0]) - t * self._dir[0]) / self._WPP + vw / 2.0,
            (float(world[1]) - t * self._dir[1]) / self._WPP + vh / 2.0,
        )

    def add_layer(self, layer):
        self.layers[layer.layer_id] = layer
        return _Handle(layer.layer_id)

    def update_layer(self, handle, layer):
        self.layers[handle.layer_id] = layer

    def remove_layer(self, handle):
        self.layers.pop(handle.layer_id, None)

    def set_layer_visible(self, handle, visible):
        handle.visible = bool(visible)

    def set_clip_planes(self, planes):
        self.clip_planes = tuple(planes or ())

    def slice_layer(self, handle, plane, *, layer_id, trim=()):
        self.slices += 1
        return MeshLayer(
            layer_id=layer_id,
            points=PointSet(np.zeros((3, 3))),
            cells=CellBlocks({"triangle": np.array([[0, 1, 2]])}),
        )


class _Iren:
    """Scriptable VTK interactor (the ``test_clip_gizmo`` rig)."""

    def __init__(self) -> None:
        self.observers: dict = {}
        self.commands: dict = {}
        self._next = 1
        self.pos = (0, 0)
        self.shift = False

    def AddObserver(self, event, cb, priority=0.0):    # noqa: N802
        tag = self._next
        self._next += 1
        self.observers[tag] = (event, cb, priority)
        self.commands[tag] = type("C", (), {
            "aborted": False,
            "SetAbortFlag": lambda self, flag: None,
        })()
        return tag

    def RemoveObserver(self, tag):                     # noqa: N802
        self.observers.pop(tag, None)

    def GetCommand(self, tag):                         # noqa: N802
        return self.commands.get(tag)

    def GetEventPosition(self):                        # noqa: N802
        return self.pos

    def GetShiftKey(self):                             # noqa: N802
        return self.shift

    def send(self, event, pos=None):
        if pos is not None:
            self.pos = pos
        for _tag, (name, cb, _p) in sorted(
            self.observers.items(), key=lambda kv: -kv[1][2],
        ):
            if name == event:
                cb(self, event)


class _Dispatcher:
    """Fires CLIP_PLANES_CHANGED at the subscribers, like the real one."""

    def __init__(self) -> None:
        self.subs: list = []

    def fire(self, kind, **_kw):
        for handler in self.subs:
            handler(kind, None)


def test_a_drag_does_not_slice_until_it_settles():
    """S2 measured a zero-actor-rebuild drag; slicing every dataset on
    every mouse-move would undo exactly that (ADR 0083 alternative 2).
    The caps are hidden mid-gesture rather than left where the plane
    used to be, and recomputed once when the user lets go."""
    backend = _CountingBackend()
    controller = ClipPlaneSetController(backend)
    controller.set_cut_face(True)
    plane = controller.add((1.0, 0.0, 0.0), offset=0.0)
    gizmos = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    gizmos.refresh()

    cut_faces = ClipCutFaceRenderer(
        backend, controller,
        sources=_sources_for("contour", _Handle("contour"), ColorSpec()),
        settled=lambda: not interactor.is_dragging,
    )
    interactor = install_clip_gizmo_interactor(backend, controller, gizmos)
    assert interactor is not None
    interactor.on_drag_end = lambda: cut_faces.refresh(force=True)

    dispatcher = _Dispatcher()
    dispatcher.subs.append(
        lambda _kind, _payload: cut_faces.refresh(),
    )
    controller.dispatcher = dispatcher

    cut_faces.refresh()
    assert backend.slices == 1
    cap = cut_faces._handles[
        f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"
    ]

    centre = np.asarray(gizmos.geometries()[plane.plane_id].centre)
    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(centre))
    for i in range(1, 21):
        backend.iren.send(
            "MouseMoveEvent",
            backend.pixel_for(centre + np.array([i * 0.02, 0.0, 0.0])),
        )
    assert controller.plane(plane.plane_id).offset == pytest.approx(0.4)
    assert backend.slices == 1, "a drag must not re-slice, tick by tick"
    assert cap.visible is False, (
        "a cap left where the plane used to be floats inside the model"
    )

    backend.iren.send("LeftButtonReleaseEvent")
    assert backend.slices == 2, "the release recomputes exactly once"
    assert cut_faces._handles[
        f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"
    ].visible is True

    # And an ordinary (non-drag) plane change still recomputes at once.
    controller.set_offset(plane.plane_id, 0.6)
    assert backend.slices == 3


def test_a_step_change_recomputes_the_cut_face():
    """The contour's dataset is re-scattered in place on every step, so
    a cap sliced at step 0 is stale at step N. The viewer drives this
    off STEP_CHANGED; here it is the same call the subscriber makes."""
    backend = _CountingBackend()
    controller = ClipPlaneSetController(backend)
    controller.set_cut_face(True)
    controller.add((1.0, 0.0, 0.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        backend, controller,
        sources=_sources_for("contour", _Handle("contour"), ColorSpec()),
        settled=lambda: True,
    )
    cut_faces.refresh()
    assert backend.slices == 1
    cut_faces.refresh()
    assert backend.slices == 2


def test_a_leave_event_settles_the_drag_too():
    """The 0081 L2 lesson applies to the cap as well: a release outside
    the window never arrives, and a drag with no end leaves the caps
    hidden forever."""
    backend = _CountingBackend()
    controller = ClipPlaneSetController(backend)
    controller.set_cut_face(True)
    plane = controller.add((1.0, 0.0, 0.0), offset=0.0)
    gizmos = ClipGizmoRenderer(backend, controller, bbox=_BBOX)
    gizmos.refresh()
    cut_faces = ClipCutFaceRenderer(
        backend, controller,
        sources=_sources_for("contour", _Handle("contour"), ColorSpec()),
        settled=lambda: not interactor.is_dragging,
    )
    interactor = install_clip_gizmo_interactor(backend, controller, gizmos)
    interactor.on_drag_end = lambda: cut_faces.refresh(force=True)
    cut_faces.refresh()

    centre = np.asarray(gizmos.geometries()[plane.plane_id].centre)
    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(centre))
    assert interactor.is_dragging is True
    backend.iren.send("LeaveEvent")
    assert interactor.is_dragging is False
    assert cut_faces._handles[
        f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:contour"
    ].visible is True


# =====================================================================
# F-TOGGLE — the panel checkbox, and what it leaves alone
# =====================================================================

@pytest.fixture
def panel():
    from qtpy import QtWidgets

    from apeGmsh.viewers.ui._clip_planes_panel import ClipPlanesPanel
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    return ClipPlanesPanel()


def test_the_checkbox_drives_the_cut_face_and_leaves_the_cut_alone(
    cut_face_backend, panel,
):
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    panel.bind(controller, bbox=_BBOX)
    assert panel._cb_cut_face.isEnabled() is True
    assert panel._cb_cut_face.isChecked() is False

    panel._cb_cut_face.setChecked(True)
    assert controller.cut_face is True
    cut_faces.refresh()
    assert len(cut_faces.layer_ids()) == 1
    assert _hot(_frame(cut_face_backend)) > 200

    panel._cb_cut_face.setChecked(False)
    assert controller.cut_face is False
    cut_faces.refresh()
    assert cut_faces.layer_ids() == set()
    assert _hot(_frame(cut_face_backend)) == 0
    # Off is off for the cap only — the geometry is still cut.
    assert _n_planes(handle) == 1


def test_apply_cuts_off_takes_the_cut_face_with_it(cut_face_backend):
    """No cut, no cut face: a cap with nothing cut open would be a
    contoured sheet floating through the middle of the model."""
    layer = _solid_layer()
    handle = cut_face_backend.add_layer(layer)
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller,
        sources=_sources_for("contour", handle, layer.color),
    )
    cut_faces.refresh()
    assert len(cut_faces.layer_ids()) == 1

    controller.set_apply_cuts(False)
    cut_faces.refresh()
    assert cut_faces.layer_ids() == set()


def test_no_contour_means_no_cut_face(cut_face_backend):
    """The cap mirrors a contour; with nothing to mirror there is
    nothing to draw (and nothing to guess a colour scale from)."""
    cut_face_backend.add_layer(_solid_layer())
    controller = ClipPlaneSetController(cut_face_backend)
    controller.set_cut_face(True)
    controller.add((0.0, 0.0, -1.0), offset=0.0)
    cut_faces = ClipCutFaceRenderer(
        cut_face_backend, controller, sources=lambda: [],
    )
    cut_faces.refresh()
    assert cut_faces.layer_ids() == set()
    assert _hot(_frame(cut_face_backend)) == 0


def test_a_hidden_contour_offers_no_cut_face_source():
    """``cut_face_source`` is the contour's own gate — a cap must never
    outlive the diagram it mirrors."""
    from apeGmsh.viewers.diagrams._contour import ContourDiagram

    diagram = ContourDiagram.__new__(ContourDiagram)
    diagram._handle = object()
    diagram._layer = _solid_layer()
    diagram._visible = True
    assert diagram.cut_face_source()[2] is diagram._layer.color

    diagram._visible = False
    assert diagram.cut_face_source() is None
    diagram._visible = True
    diagram._handle = None
    assert diagram.cut_face_source() is None


# =====================================================================
# F-SESSION — the flag round-trips, and v9 loads clean
# =====================================================================

def test_the_cut_face_flag_round_trips_through_a_session(tmp_path):
    from apeGmsh.viewers.diagrams._session import (
        SESSION_SCHEMA_VERSION,
        deserialize_session,
        load_session,
        save_session,
    )

    assert SESSION_SCHEMA_VERSION == 10

    path = save_session(
        specs=[], results_path=tmp_path / "r.h5", fem_snapshot_id=None,
        target_path=tmp_path / "s.json", clip_cut_face=True,
    )
    assert load_session(path).clip_cut_face is True

    # A v9 session predates the field entirely and must load as "off",
    # which is what a fresh viewer starts at.
    legacy = deserialize_session({
        "schema_version": 9,
        "results_path": str(tmp_path / "r.h5"),
        "diagrams": [],
        "clip_planes": [],
        "clip_apply_cuts": True,
        "clip_show_gizmos": True,
    })
    assert legacy.clip_cut_face is False


def test_the_controller_restores_the_flag():
    controller = ClipPlaneSetController(None)
    controller.restore([], cut_face=True)
    assert controller.cut_face is True
    controller.restore([])
    assert controller.cut_face is False, "a legacy restore reads as off"
