"""Section planes — the cut in the picture, not the call (ADR 0083 S1).

The ADR's lesson from 0081 is that asserting "we called VTK" keeps a
suite green through every defect a user reports, so these assert what
lands on screen and what the user can then click:

* **C-CUT** — geometry is gone on the cut side and present on the kept
  side; flip inverts it.
* **C-REBUILD** — a glyph layer stepping through ``update_layer``'s
  rebuild path keeps the cut (probe P1's regression test).
* **C-SUBSTRATE** — off-seam actors created while a cut is live arrive
  clipped (probe P2).
* **C-PICK** — a click on a clipped-away cell is a miss; the same click
  with cuts off resolves (probe P3).
* **C-DEFORM** — the cut tracks the warp: a fixed plane clips a
  different set of pixels at step 0 and step N.
* **C-SESSION** — planes round-trip; a v8 session loads with none.
* **C-EXPORT** — exported frames carry the cut (probe P6's pin).
* **C-LIMIT** — the seventh cutting plane is refused; six work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.viewers.core._clip_planes import (
    MAX_ACTIVE_PLANES,
    ClipPlaneSetController,
)
from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ClipPlaneSpec,
    ColorSpec,
    GlyphLayer,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarField,
)


# =====================================================================
# Helpers
# =====================================================================

#: Anything brighter than this on a black background is painted
#: geometry. Well below the darkest end of any colormap ramp.
_PAINTED = 12


@pytest.fixture
def cut_backend():
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


def _brick_layer(layer_id: str = "brick", *, warp: float = 0.0) -> MeshLayer:
    """One contoured hexahedron spanning ``[-1, 1]`` on every axis.

    ``warp`` displaces the ``+x`` face outward, standing in for a
    deformation step: the *rendered* shape moves while the plane does
    not.
    """
    pts = np.array(
        [
            [-1, -1, -1], [1 + warp, -1, -1], [1 + warp, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1 + warp, -1, 1], [1 + warp, 1, 1], [-1, 1, 1],
        ],
        dtype=float,
    )
    return MeshLayer(
        layer_id=layer_id,
        points=PointSet(pts),
        cells=CellBlocks({"hexahedron": np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}),
        fields=(
            ScalarField(name="v", values=pts[:, 2].copy(), location="point"),
        ),
        color=ColorSpec(
            mode="by_array", array_name="v",
            lut=LutSpec(name="jet", vmin=-1.0, vmax=1.0),
        ),
    )


def _pin_camera(backend) -> None:
    """Fix the view so image-left always means world ``x < 0``.

    Deliberately not ``reset_camera()``: clipping and deformation both
    change the scene bounds, and a camera that refits would move the
    plane's image column between the two frames a test is comparing.
    """
    camera = backend.plotter.camera
    camera.parallel_projection = True
    camera.position = (0.0, 0.0, 10.0)
    camera.focal_point = (0.0, 0.0, 0.0)
    camera.up = (0.0, 1.0, 0.0)
    camera.parallel_scale = 3.0
    camera.clipping_range = (0.1, 100.0)


def _frame(backend) -> np.ndarray:
    """Render and return the current frame as ``(h, w, 3)`` uint8."""
    _pin_camera(backend)
    plotter = backend.plotter
    plotter.render()
    return np.asarray(
        plotter.screenshot(return_img=True, transparent_background=False),
    )


def _halves(img: np.ndarray) -> tuple[int, int]:
    """``(painted left of centre, painted right of centre)``."""
    painted = img.astype(np.int32).sum(axis=2) > _PAINTED
    mid = painted.shape[1] // 2
    return int(painted[:, :mid].sum()), int(painted[:, mid:].sum())


def _n_planes(handle) -> int:
    """Clip planes carried by a handle's (or actor's) mapper."""
    actor = getattr(handle, "actor", handle)
    return int(actor.GetMapper().GetNumberOfClippingPlanes())


# =====================================================================
# C-CUT — the cut side is gone, the kept side is not
# =====================================================================

def test_cut_removes_the_discarded_half(cut_backend):
    """A plane at the origin halves the brick, and flip swaps which
    half survives. The whole feature in one assertion set."""
    handle = cut_backend.add_layer(_brick_layer())
    left_full, right_full = _halves(_frame(cut_backend))
    assert left_full > 0 and right_full > 0, "brick must render uncut"

    # +X normal: the +x half survives.
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )
    left, right = _halves(_frame(cut_backend))
    assert left == 0, "geometry on the cut side must be gone"
    assert right == pytest.approx(right_full, rel=0.05)

    # Flipped: the -x half survives instead.
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))],
    )
    left, right = _halves(_frame(cut_backend))
    assert right == 0
    assert left == pytest.approx(left_full, rel=0.05)

    # Empty set restores the whole brick.
    cut_backend.set_clip_planes([])
    assert _halves(_frame(cut_backend)) == (left_full, right_full)
    assert _n_planes(handle) == 0


def test_labels_are_never_cut(cut_backend):
    """2D text mappers do not take world clip planes (probe P8)."""
    from apeGmsh.viewers.scene_ir import LabelLayer

    handle = cut_backend.add_layer(
        LabelLayer(
            layer_id="lbl",
            positions=PointSet(np.array([[-1.0, 0.0, 0.0]])),
            texts=("n1",),
        ),
    )
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )
    assert handle.kind == "label"
    assert _n_planes(handle) == 0


# =====================================================================
# C-REBUILD — the rebuild path keeps the cut (probe P1)
# =====================================================================

def _glyph_layer(x_shift: float = 0.0, n: int = 6) -> GlyphLayer:
    xs = np.linspace(-1.0, 1.0, n) + x_shift
    pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    return GlyphLayer(
        layer_id="glyphs",
        positions=PointSet(pts),
        kind="sphere",
        color=ColorSpec(mode="solid", solid_rgb=(1.0, 1.0, 1.0)),
    )


def test_glyph_step_keeps_the_cut(cut_backend):
    """A glyph diagram animating through a live cut stays cut.

    ``update_layer`` has no in-place path for glyphs — it destroys and
    re-creates the actor and its mapper on every step. Attaching the
    planes once to the live mappers would leave the animation running
    uncut from step 1 onward, which is the defect probe P1 found.
    """
    handle = cut_backend.add_layer(_glyph_layer())
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )
    left_before, right_before = _halves(_frame(cut_backend))
    assert left_before == 0 and right_before > 0

    cut_backend.update_layer(handle, _glyph_layer(x_shift=0.05))

    assert _n_planes(handle) == 1, "the re-created mapper lost the cut"
    left_after, right_after = _halves(_frame(cut_backend))
    assert left_after == 0, "the step re-rendered the clipped-away half"
    assert right_after > 0


def test_a_layer_added_after_the_cut_arrives_clipped(cut_backend):
    """Stamping, not attaching: order of operations must not matter."""
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )
    handle = cut_backend.add_layer(_brick_layer())
    assert _n_planes(handle) == 1
    left, right = _halves(_frame(cut_backend))
    assert left == 0 and right > 0


def test_removed_layers_leave_the_registry(cut_backend):
    """A dropped layer must not keep its actor alive in the registry."""
    handle = cut_backend.add_layer(_brick_layer())
    assert "brick" in cut_backend._handles
    cut_backend.remove_layer(handle)
    assert "brick" not in cut_backend._handles
    # Re-cutting after the removal must not raise on the dead handle.
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )


# =====================================================================
# C-SUBSTRATE — off-seam actors arrive clipped (probe P2)
# =====================================================================

class _Palette:
    substrate_color = "#808080"
    substrate_edge_color = "#c0c0c0"
    node_accent = "#ff8800"


class _Prefs:
    mesh_surface_opacity = 1.0
    mesh_line_width = 1.0
    point_size = 6.0


class _Scene:
    """Just enough ``FEMSceneData`` for the off-seam actor builders."""

    def __init__(self) -> None:
        import pyvista as pv

        self.grid = pv.Box(bounds=(-1, 1, -1, 1, -1, 1)).cast_to_unstructured_grid()
        self.model_diagonal = 3.46


def test_substrate_and_node_cloud_are_cut_at_creation(cut_backend):
    """A geometry materialized *after* the cut renders clipped.

    ``_add_substrate_actors`` runs once at boot and again per
    materialized geometry, so a cut set up before a second geometry
    appears has to reach the later pair — which is why the attachment
    lives inside the builder rather than at the boot call site.
    """
    from apeGmsh.viewers.results_viewer import (
        add_node_cloud_actor, add_substrate_actors,
    )

    plotter = cut_backend.plotter
    scene = _Scene()
    specs = [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))]

    fill, wf = add_substrate_actors(
        plotter, scene, palette=_Palette(), prefs=_Prefs(),
        clip_planes=(), name_suffix="@first",
    )
    assert _n_planes(fill) == 0 and _n_planes(wf) == 0

    fill2, wf2 = add_substrate_actors(
        plotter, scene, palette=_Palette(), prefs=_Prefs(),
        clip_planes=specs, name_suffix="@second",
    )
    assert _n_planes(fill2) == 1 and _n_planes(wf2) == 1

    cloud = add_node_cloud_actor(
        plotter, scene, marker_size=6.0, color="#ff8800",
        clip_planes=specs,
    )
    assert cloud is not None and _n_planes(cloud) == 1


def test_substrate_pair_renders_only_the_kept_half(cut_backend):
    """The pixel half of C-SUBSTRATE — the actors really are cut."""
    from apeGmsh.viewers.results_viewer import add_substrate_actors

    add_substrate_actors(
        cut_backend.plotter, _Scene(), palette=_Palette(), prefs=_Prefs(),
        clip_planes=[
            ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
        ],
    )
    left, right = _halves(_frame(cut_backend))
    assert left == 0 and right > 0


# =====================================================================
# C-PICK — a hit the cut has hidden is a miss (probe P3)
# =====================================================================

class _FakePickBackend:
    """Captures the callbacks ``install_results_pick`` registers."""

    def __init__(self) -> None:
        self.on_pick = None

    def install(self, *, on_pick, on_box) -> None:
        self.on_pick = on_pick

    def project_points(self, points):
        return np.zeros((len(points), 2))

    def uninstall(self) -> None:
        pass


class _PickScene:
    pick_engine = None
    cell_dim = np.array([], dtype=np.int8)
    cell_to_element_id = np.array([7], dtype=np.int64)
    element_id_to_cell = {7: 0}
    node_ids = np.array([1], dtype=np.int64)


def _pick_once(controller, world):
    """Drive one click at ``world``; return the PickResult or ``None``."""
    from apeGmsh.viewers.core.results_pick import install_results_pick
    from apeGmsh.viewers.scene_ir import PickHit

    backend = _FakePickBackend()
    seen: list = []
    install_results_pick(
        None, seen.append, scene=_PickScene(),
        pick_backend=backend, clip_planes=controller,
    )
    backend.on_pick(PickHit(world=world, cell_id=0, prop_id=1), None)
    return seen[0] if seen else None


def test_a_click_on_a_clipped_away_cell_is_a_miss():
    """``vtkCellPicker`` is blind to mapper clip planes — the
    controller is what stops a phantom pick on empty-looking space."""
    controller = ClipPlaneSetController(None)
    controller.add((1.0, 0.0, 0.0), offset=0.0)

    assert _pick_once(controller, (-0.5, 0.0, 0.0)) is None
    kept = _pick_once(controller, (0.5, 0.0, 0.0))
    assert kept is not None and kept.kind == "node"


def test_the_same_click_resolves_with_cuts_off():
    """Turning the master toggle off restores the ordinary pick."""
    controller = ClipPlaneSetController(None)
    controller.add((1.0, 0.0, 0.0), offset=0.0)
    controller.set_apply_cuts(False)

    result = _pick_once(controller, (-0.5, 0.0, 0.0))
    assert result is not None and result.kind == "node"


def test_a_point_on_the_cut_face_is_not_rejected():
    """Exactly on the plane is kept — the cut face is visible geometry."""
    controller = ClipPlaneSetController(None)
    controller.add((1.0, 0.0, 0.0), offset=0.25)
    assert controller.is_clipped((0.25, 0.0, 0.0)) is False


# =====================================================================
# C-DEFORM — the cut tracks the warp
# =====================================================================

def test_the_cut_follows_the_deformed_shape(cut_backend):
    """A world-fixed plane cuts what is *rendered*, so the set of
    clipped pixels changes as the shape warps through it."""
    handle = cut_backend.add_layer(_brick_layer())
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )
    step0 = _halves(_frame(cut_backend))

    # Step N: the +x face has moved further through the plane.
    cut_backend.update_layer(handle, _brick_layer(warp=1.5))
    stepn = _halves(_frame(cut_backend))

    assert stepn != step0, "the cut did not track the deformation"
    assert stepn[1] > step0[1], "more of the shape crossed the plane"


# =====================================================================
# C-SESSION — planes survive save / load
# =====================================================================

def test_planes_round_trip_through_a_session(tmp_path: Path):
    from apeGmsh.viewers.diagrams._session import (
        ClipPlaneSnapshot, load_session, save_session,
    )

    controller = ClipPlaneSetController(None)
    first = controller.add((1.0, 0.0, 0.0), offset=0.25)
    second = controller.add((0.0, 0.0, 1.0), offset=-2.0)
    controller.set_flipped(second.plane_id, True)
    controller.set_active(second.plane_id, False)

    path = save_session(
        specs=[], results_path=tmp_path / "run.h5", fem_snapshot_id=None,
        clip_planes=[
            ClipPlaneSnapshot(**raw) for raw in controller.snapshot()
        ],
        clip_apply_cuts=controller.apply_cuts,
        clip_show_gizmos=False,
    )
    session = load_session(path)

    assert len(session.clip_planes) == 2
    assert session.clip_show_gizmos is False

    restored = ClipPlaneSetController(None)
    restored.restore(
        [
            {
                "plane_id": cp.plane_id, "name": cp.name,
                "normal": cp.normal, "offset": cp.offset,
                "active": cp.active, "flipped": cp.flipped,
                "gizmo_visible": cp.gizmo_visible,
            }
            for cp in session.clip_planes
        ],
        apply_cuts=session.clip_apply_cuts,
        show_gizmos=session.clip_show_gizmos,
    )
    planes = restored.planes()
    assert [p.plane_id for p in planes] == [first.plane_id, second.plane_id]
    assert planes[0].active is True
    assert planes[0].offset == pytest.approx(0.25)
    assert planes[1].active is False
    assert planes[1].flipped is True
    assert restored.show_gizmos is False
    # Only the active plane cuts.
    assert len(restored.specs()) == 1


def test_a_v8_session_loads_with_no_planes(tmp_path: Path):
    """A session written before ADR 0083 has no ``clip_planes`` key."""
    import json

    from apeGmsh.viewers.diagrams._session import load_session

    path = tmp_path / "legacy.viewer-session.json"
    path.write_text(
        json.dumps({
            "schema_version": 8,
            "results_path": str(tmp_path / "run.h5"),
            "fem_snapshot_id": None,
            "saved_at": "2026-01-01T00:00:00+00:00",
            "diagrams": [],
        }),
        encoding="utf-8",
    )
    session = load_session(path)
    assert session.clip_planes == ()
    assert session.clip_apply_cuts is True

    controller = ClipPlaneSetController(None)
    controller.restore(session.clip_planes)
    assert controller.planes() == []
    assert controller.specs() == ()


def test_a_plane_added_after_a_sparse_restore_gets_a_fresh_id():
    """Adversarial-review regression (A1): a session saved after a
    deletion restores sparse ids (plane-1, plane-3); the next ``add``
    must not mint plane-3 again — with a duplicate id, every id-keyed
    mutator drives the first match, i.e. the wrong plane."""
    controller = ClipPlaneSetController(None)
    controller.restore([
        {"plane_id": "plane-1", "normal": (1, 0, 0)},
        {"plane_id": "plane-3", "normal": (0, 1, 0)},
    ])
    new = controller.add((0, 0, 1))
    ids = [p.plane_id for p in controller.planes()]
    assert len(ids) == len(set(ids)), f"duplicate plane id: {ids}"
    controller.set_offset(new.plane_id, 5.0)
    assert new.offset == 5.0
    assert controller.plane("plane-3").offset == 0.0


# =====================================================================
# C-EXPORT — exported frames carry the cut (probe P6)
# =====================================================================

class _StubDirector:
    """The three attributes ``export_animation`` reads."""

    def __init__(self, n_steps: int = 3) -> None:
        self.n_steps = n_steps
        self.step_index = 0

    def set_step(self, index: int) -> None:
        self.step_index = int(index)


def test_exported_frames_show_the_cut(cut_backend, tmp_path: Path):
    """``export_animation`` reuses the live plotter, so whatever the
    mappers carry is what gets encoded — pinned rather than assumed."""
    pytest.importorskip("imageio")
    import imageio.v2 as imageio

    from apeGmsh.viewers.animation import export_animation

    # The handle is held for the duration, as a diagram holds its own:
    # the backend's registry is weak by design (ADR 0083 Part 2).
    handle = cut_backend.add_layer(_brick_layer())
    cut_backend.set_clip_planes(
        [ClipPlaneSpec(origin=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))],
    )
    assert _n_planes(handle) == 1
    _frame(cut_backend)     # frame the camera the way the viewer would

    out = export_animation(
        cut_backend.plotter, _StubDirector(), tmp_path / "cut.gif", fps=5,
    )
    frames = imageio.mimread(out)
    assert frames
    for frame in frames:
        left, right = _halves(np.asarray(frame)[:, :, :3])
        assert left == 0, "an exported frame lost the cut"
        assert right > 0


# =====================================================================
# C-LIMIT — six planes cut, the seventh is refused
# =====================================================================

def test_the_seventh_active_plane_is_refused():
    """OpenGL guarantees six user clip distances; a silently dropped
    seventh would make the panel lie about what is cutting."""
    refusals: list[str] = []
    controller = ClipPlaneSetController(None)
    controller.on_status = refusals.append

    made = [
        controller.add((1.0, 0.0, 0.0), offset=float(i))
        for i in range(MAX_ACTIVE_PLANES)
    ]
    assert all(p.active for p in made)
    assert len(controller.specs()) == MAX_ACTIVE_PLANES
    assert refusals == []

    seventh = controller.add((0.0, 1.0, 0.0))
    assert seventh.active is False, "a seventh plane must not cut"
    assert len(controller.specs()) == MAX_ACTIVE_PLANES
    assert len(refusals) == 1

    assert controller.set_active(seventh.plane_id, True) is False
    assert len(refusals) == 2

    # Free a slot and the seventh can take it.
    controller.set_active(made[0].plane_id, False)
    assert controller.set_active(seventh.plane_id, True) is True
    assert len(controller.specs()) == MAX_ACTIVE_PLANES


def test_a_restored_session_cannot_exceed_the_cap():
    controller = ClipPlaneSetController(None)
    controller.restore([
        {
            "plane_id": f"plane-{i}", "name": f"Plane {i}",
            "normal": (1.0, 0.0, 0.0), "offset": float(i), "active": True,
        }
        for i in range(MAX_ACTIVE_PLANES + 2)
    ])
    assert len(controller.planes()) == MAX_ACTIVE_PLANES + 2
    assert len(controller.specs()) == MAX_ACTIVE_PLANES


# =====================================================================
# Owner contract (ADR 0056 Part 2) — one event per mutation
# =====================================================================

class _RecordingDispatcher:
    def __init__(self) -> None:
        self.fired: list[str] = []

    def fire(self, kind: str, **_kw) -> None:
        self.fired.append(kind)


def test_every_mutator_fires_exactly_one_event():
    from apeGmsh.viewers.diagrams._dispatch import CLIP_PLANES_CHANGED

    dispatcher = _RecordingDispatcher()
    controller = ClipPlaneSetController(None)
    controller.dispatcher = dispatcher

    plane = controller.add((1.0, 0.0, 0.0))
    controller.set_offset(plane.plane_id, 0.5)
    controller.set_flipped(plane.plane_id, True)
    controller.set_normal(plane.plane_id, (0.0, 1.0, 0.0))
    controller.set_active(plane.plane_id, False)
    controller.set_apply_cuts(False)
    controller.set_show_gizmos(False)
    controller.remove(plane.plane_id)

    assert dispatcher.fired == [CLIP_PLANES_CHANGED] * 8

    # A no-op mutation is silent — the dispatcher's coalesced render is
    # not free, and a panel echoing state back must not cost a frame.
    dispatcher.fired.clear()
    controller.set_apply_cuts(False)
    controller.set_offset("no-such-plane", 1.0)
    assert dispatcher.fired == []


def test_apply_cuts_off_leaves_the_planes_defined():
    controller = ClipPlaneSetController(None)
    controller.add((1.0, 0.0, 0.0), offset=0.5)
    controller.set_apply_cuts(False)
    assert controller.specs() == ()
    assert len(controller.planes()) == 1
    controller.set_apply_cuts(True)
    assert len(controller.specs()) == 1


def test_the_controller_pushes_its_set_to_the_backend(backend):
    """The reconcile half of the owner contract."""
    controller = ClipPlaneSetController(backend)
    controller.add((0.0, 0.0, 1.0), offset=2.0)
    assert backend.clip_planes == (
        ClipPlaneSpec(origin=(0.0, 0.0, 2.0), normal=(0.0, 0.0, 1.0)),
    )
    controller.clear()
    assert backend.clip_planes == ()


# =====================================================================
# The panel is a projection, not a second copy of the state
# =====================================================================

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("qtpy.QtCore")
    from qtpy import QtWidgets
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp):
    from apeGmsh.viewers.ui._clip_planes_panel import ClipPlanesPanel
    return ClipPlanesPanel()


def test_panel_drives_the_controller(panel):
    controller = ClipPlaneSetController(None)
    panel.bind(controller, bbox=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))

    panel._combo_add.setCurrentText("Z")
    panel._on_add()
    planes = controller.planes()
    assert len(planes) == 1
    assert planes[0].normal == (0.0, 0.0, 1.0)
    assert panel.selected_id == planes[0].plane_id

    # The slider spans the bbox along the plane normal, not a raw
    # 0..1 range — the far end must land on the far face.
    panel._slider.setValue(1000)
    assert controller.plane(panel.selected_id).offset == pytest.approx(1.0)

    # The numeric field is unclamped: deformation can push geometry
    # well outside the reference bbox the slider maps.
    panel._spin_offset.setValue(250.0)
    assert controller.plane(panel.selected_id).offset == pytest.approx(250.0)

    panel._on_flip(panel.selected_id, True)
    assert controller.plane(panel.selected_id).flipped is True

    panel._on_delete(panel.selected_id)
    assert controller.planes() == []


def test_panel_refuses_to_show_a_seventh_cut(panel):
    """A refused activation leaves the checkbox where the scene is."""
    controller = ClipPlaneSetController(None)
    panel.bind(controller)
    for _ in range(MAX_ACTIVE_PLANES + 1):
        panel._on_add()

    seventh = controller.planes()[-1]
    assert seventh.active is False
    panel._on_active(seventh.plane_id, True)
    assert controller.plane(seventh.plane_id).active is False
    assert len(controller.specs()) == MAX_ACTIVE_PLANES


def test_gizmo_eye_is_present_but_inert(panel):
    """Slice 2 draws the gizmo; the control must not pretend otherwise."""
    from qtpy import QtWidgets

    controller = ClipPlaneSetController(None)
    panel.bind(controller)
    panel._on_add()

    row = panel._list_layout.itemAt(0).widget()
    buttons = row.findChildren(QtWidgets.QToolButton)
    eyes = [b for b in buttons if b.text() == "👁"]
    assert len(eyes) == 1
    assert eyes[0].isEnabled() is False
    assert panel._cb_cut_face.isEnabled() is False
