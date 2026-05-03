"""install_results_pick — observer install + click-pick filtering.

Same pattern as the (deleted) ShiftClickPicker tests: simulate VTK
events with stub callers and verify the right observer paths fire.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def _stub_caller(*, shift: bool = False, x: int = 100, y: int = 100):
    caller = MagicMock()
    caller.GetShiftKey.return_value = 1 if shift else 0
    caller.GetEventPosition.return_value = (x, y)
    return caller


def _stub_plotter():
    iren = MagicMock()
    iren.AddObserver.side_effect = [101, 102, 103]
    plotter = MagicMock()
    plotter.iren = iren
    plotter.iren.interactor = iren
    plotter.renderer = MagicMock()
    return plotter, iren


def _stub_scene(cell_to_element_id=None):
    """Minimal FEMSceneData stand-in with the cell→element-id map."""
    if cell_to_element_id is None:
        cell_to_element_id = np.array([1001, 1002, 1003], dtype=np.int64)
    scene = MagicMock()
    scene.cell_to_element_id = np.asarray(cell_to_element_id, dtype=np.int64)
    return scene


def _capture_handlers(iren):
    return {
        call.args[0]: call.args[1]
        for call in iren.AddObserver.call_args_list
    }


def _install(plotter, on_pick=None, scene=None):
    from apeGmsh.viewers.core.results_pick import install_results_pick
    seen = []
    cb = on_pick if on_pick is not None else (lambda r: seen.append(r))
    ctrl = install_results_pick(
        plotter, cb, scene=scene if scene is not None else _stub_scene(),
    )
    return seen, ctrl


def _patch_picker(monkeypatch, *, cell_id: int, world=(0.0, 0.0, 0.0)):
    fake_picker = MagicMock()
    fake_picker.GetCellId.return_value = cell_id
    fake_picker.GetPickPosition.return_value = world
    import vtk
    monkeypatch.setattr(vtk, "vtkCellPicker", lambda: fake_picker)
    return fake_picker


def _click(handlers, x: int = 50, y: int = 60, shift: bool = False) -> None:
    handlers["LeftButtonPressEvent"](
        _stub_caller(x=x, y=y, shift=shift), "LeftButtonPressEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=x, y=y, shift=shift), "LeftButtonReleaseEvent",
    )


# =====================================================================
# Default mode is node; click on hit fires a node PickResult.
# =====================================================================

def test_default_mode_is_node(monkeypatch):
    _patch_picker(monkeypatch, cell_id=42, world=(1.0, 2.0, 3.0))
    plotter, iren = _stub_plotter()
    seen, ctrl = _install(plotter)
    handlers = _capture_handlers(iren)
    assert ctrl.mode == "node"

    _click(handlers)

    assert len(seen) == 1
    r = seen[0]
    assert r.kind == "node"
    assert r.world == (1.0, 2.0, 3.0)
    assert r.element_id is None


# =====================================================================
# Element mode resolves cell_id → FEM element_id via scene.
# =====================================================================

def test_element_mode_resolves_to_element_id(monkeypatch):
    _patch_picker(monkeypatch, cell_id=1, world=(0.5, 0.5, 0.0))
    plotter, iren = _stub_plotter()
    scene = _stub_scene(cell_to_element_id=[1001, 1002, 1003])
    seen, ctrl = _install(plotter, scene=scene)
    handlers = _capture_handlers(iren)
    ctrl.set_mode("element")

    _click(handlers)

    assert len(seen) == 1
    r = seen[0]
    assert r.kind == "element"
    assert r.element_id == 1002
    assert r.cell_id == 1
    assert r.world == (0.5, 0.5, 0.0)


# =====================================================================
# Element mode: out-of-range cell_id (e.g. probe-marker actor) is
# filtered out so we don't index past the substrate map.
# =====================================================================

def test_element_mode_drops_out_of_range_cell(monkeypatch):
    _patch_picker(monkeypatch, cell_id=999, world=(0.0, 0.0, 0.0))
    plotter, iren = _stub_plotter()
    scene = _stub_scene(cell_to_element_id=[1001, 1002, 1003])
    seen, ctrl = _install(plotter, scene=scene)
    handlers = _capture_handlers(iren)
    ctrl.set_mode("element")

    _click(handlers)
    assert seen == []


# =====================================================================
# Miss (cell_id < 0) suppresses the callback regardless of mode.
# =====================================================================

@pytest.mark.parametrize("mode", ["node", "element"])
def test_miss_does_not_invoke_callback(monkeypatch, mode):
    _patch_picker(monkeypatch, cell_id=-1)
    plotter, iren = _stub_plotter()
    seen, ctrl = _install(plotter)
    ctrl.set_mode(mode)
    handlers = _capture_handlers(iren)

    _click(handlers)
    assert seen == []


# =====================================================================
# Shift+LMB is owned by navigation — pick observer must bail.
# =====================================================================

def test_shift_lmb_press_does_not_arm_pick(monkeypatch):
    _patch_picker(monkeypatch, cell_id=7)
    plotter, iren = _stub_plotter()
    seen, _ = _install(plotter)
    handlers = _capture_handlers(iren)

    _click(handlers, shift=True)
    assert seen == []


# =====================================================================
# Drag releases do not pick (Phase 2c will handle box-select).
# =====================================================================

def test_drag_release_does_not_pick(monkeypatch):
    _patch_picker(monkeypatch, cell_id=7, world=(5.0, 5.0, 5.0))
    plotter, iren = _stub_plotter()
    seen, _ = _install(plotter)
    handlers = _capture_handlers(iren)

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=10, y=20), "LeftButtonPressEvent",
    )
    handlers["MouseMoveEvent"](
        _stub_caller(x=30, y=40), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=30, y=40), "LeftButtonReleaseEvent",
    )
    assert seen == []


# =====================================================================
# Callback exceptions are swallowed.
# =====================================================================

def test_callback_exception_is_swallowed(monkeypatch, capsys):
    _patch_picker(monkeypatch, cell_id=1)
    plotter, iren = _stub_plotter()
    from apeGmsh.viewers.core.results_pick import install_results_pick
    def _raise(_r):
        raise RuntimeError("boom")
    install_results_pick(plotter, _raise, scene=_stub_scene())
    handlers = _capture_handlers(iren)

    _click(handlers)
    captured = capsys.readouterr()
    assert "on_pick raised" in captured.err


# =====================================================================
# Controller validates mode strings.
# =====================================================================

def test_controller_rejects_invalid_mode():
    from apeGmsh.viewers.core.results_pick import ResultsPickController
    ctrl = ResultsPickController()
    with pytest.raises(ValueError, match="mode"):
        ctrl.set_mode("bogus")


# =====================================================================
# Box-pick — pure projection / inside-box helpers
# =====================================================================

def test_inside_box_basic():
    from apeGmsh.viewers.core.results_pick import _inside_box
    pts = np.array([
        [10.0, 10.0],
        [50.0, 50.0],
        [99.0, 99.0],
        [101.0, 50.0],    # outside (x too big)
        [50.0, -1.0],     # outside (y too small)
    ])
    mask = _inside_box(pts, 0.0, 0.0, 100.0, 100.0)
    np.testing.assert_array_equal(mask, [True, True, True, False, False])


def test_inside_box_handles_reversed_corners():
    """A right-to-left drag should still produce the correct mask
    (the helper sorts the corners internally before testing)."""
    from apeGmsh.viewers.core.results_pick import _inside_box
    pts = np.array([[50.0, 50.0]])
    mask_lr = _inside_box(pts, 0.0, 0.0, 100.0, 100.0)
    mask_rl = _inside_box(pts, 100.0, 100.0, 0.0, 0.0)
    np.testing.assert_array_equal(mask_lr, mask_rl)


# =====================================================================
# Drag release fires on_box_pick (mode=node).
# =====================================================================

def test_drag_release_node_mode_box_picks(monkeypatch):
    # Fake the per-point projection so the test doesn't need a live
    # renderer. We project point i to display (i*10, i*10) — first
    # three points fall inside (0,0)-(25,25), the fourth is outside.
    import apeGmsh.viewers.core.results_pick as mod

    def fake_project(points, _renderer):
        return np.column_stack([
            np.arange(points.shape[0]) * 10.0,
            np.arange(points.shape[0]) * 10.0,
        ])
    monkeypatch.setattr(mod, "_project_points_to_display", fake_project)

    plotter, iren = _stub_plotter()
    grid = MagicMock()
    grid.points = np.zeros((4, 3), dtype=np.float64)
    scene = MagicMock()
    scene.grid = grid
    scene.cell_to_element_id = np.array([1001, 1002, 1003], dtype=np.int64)
    scene.node_ids = np.array([10, 20, 30, 40], dtype=np.int64)

    box_seen: list = []
    pick_seen: list = []
    from apeGmsh.viewers.core.results_pick import install_results_pick
    install_results_pick(
        plotter,
        on_pick=pick_seen.append,
        on_box_pick=box_seen.append,
        scene=scene,
    )
    handlers = _capture_handlers(iren)

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=0, y=0), "LeftButtonPressEvent",
    )
    # Cross drag threshold so the engine flags a drag.
    handlers["MouseMoveEvent"](
        _stub_caller(x=25, y=25), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=25, y=25), "LeftButtonReleaseEvent",
    )

    assert pick_seen == []
    assert len(box_seen) == 1
    box = box_seen[0]
    assert box.kind == "node"
    np.testing.assert_array_equal(box.ids, [10, 20, 30])
    assert box.crossing is False


def test_drag_release_element_mode_box_picks(monkeypatch):
    import apeGmsh.viewers.core.results_pick as mod

    def fake_project(points, _renderer):
        return np.column_stack([
            np.arange(points.shape[0]) * 10.0,
            np.arange(points.shape[0]) * 10.0,
        ])
    monkeypatch.setattr(mod, "_project_points_to_display", fake_project)

    plotter, iren = _stub_plotter()
    grid = MagicMock()
    grid.cell_centers.return_value.points = np.zeros((4, 3), dtype=np.float64)
    grid.points = np.zeros((4, 3), dtype=np.float64)
    scene = MagicMock()
    scene.grid = grid
    scene.cell_to_element_id = np.array(
        [1001, 1002, 1003, 1004], dtype=np.int64,
    )
    scene.node_ids = np.zeros(4, dtype=np.int64)

    box_seen: list = []
    from apeGmsh.viewers.core.results_pick import install_results_pick
    ctrl = install_results_pick(
        plotter,
        on_pick=lambda r: None,
        on_box_pick=box_seen.append,
        scene=scene,
    )
    ctrl.set_mode("element")
    handlers = _capture_handlers(iren)

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=100, y=100), "LeftButtonPressEvent",
    )
    # Drag right-to-left → "crossing" mode (x1 < x0).
    handlers["MouseMoveEvent"](
        _stub_caller(x=0, y=0), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=0, y=0), "LeftButtonReleaseEvent",
    )

    assert len(box_seen) == 1
    box = box_seen[0]
    assert box.kind == "element"
    # Cells whose centroids project to (0,0)..(40,40) — all fall
    # within the (100,100)→(0,0) box, so all four are picked.
    np.testing.assert_array_equal(box.ids, [1001, 1002, 1003, 1004])
    np.testing.assert_array_equal(box.cell_ids, [0, 1, 2, 3])
    assert box.crossing is True


def test_degenerate_drag_does_not_box_pick(monkeypatch):
    """A drag that crosses the threshold but still ends with x0==x1
    or y0==y1 (zero-area rectangle) should not fire on_box_pick."""
    plotter, iren = _stub_plotter()
    scene = _stub_scene()
    box_seen: list = []
    from apeGmsh.viewers.core.results_pick import install_results_pick
    install_results_pick(
        plotter,
        on_pick=lambda r: None,
        on_box_pick=box_seen.append,
        scene=scene,
    )
    handlers = _capture_handlers(iren)

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=10, y=10), "LeftButtonPressEvent",
    )
    handlers["MouseMoveEvent"](
        _stub_caller(x=10, y=30), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=10, y=10), "LeftButtonReleaseEvent",   # x0 == x1
    )
    assert box_seen == []


# =====================================================================
# GP mode dispatch via gp_resolver.
# =====================================================================

def test_gp_mode_dispatches_via_resolver(monkeypatch):
    """When mode == "gp", the pick observer queries gp_resolver with
    the picked actor + cell_id and packages the result as a
    PickResult(kind="gp", element_id=..., gp_index=..., world=...)."""
    fake_picker = MagicMock()
    fake_picker.GetCellId.return_value = 5
    fake_picker.GetPickPosition.return_value = (0.0, 0.0, 0.0)
    fake_actor = MagicMock(name="GpDiagramActor")
    fake_picker.GetActor.return_value = fake_actor
    import vtk
    monkeypatch.setattr(vtk, "vtkCellPicker", lambda: fake_picker)

    plotter, iren = _stub_plotter()
    seen, ctrl = _install(plotter)    # default scene; not used in GP path
    handlers = _capture_handlers(iren)

    resolver_calls: list = []
    def resolver(actor, cell_id):
        resolver_calls.append((actor, cell_id))
        return (1042, 7, np.array([1.0, 2.0, 3.0]))

    # Re-install with the resolver wired in.
    plotter, iren = _stub_plotter()
    seen2 = []
    from apeGmsh.viewers.core.results_pick import install_results_pick
    ctrl2 = install_results_pick(
        plotter,
        on_pick=seen2.append,
        scene=_stub_scene(),
        gp_resolver=resolver,
    )
    handlers = _capture_handlers(iren)
    ctrl2.set_mode("gp")

    _click(handlers, x=50, y=60)

    assert len(resolver_calls) == 1
    assert resolver_calls[0] == (fake_actor, 5)
    assert len(seen2) == 1
    r = seen2[0]
    assert r.kind == "gp"
    assert r.element_id == 1042
    assert r.gp_index == 7
    assert r.world == (1.0, 2.0, 3.0)


def test_gp_mode_resolver_returning_none_skips_pick(monkeypatch):
    """If the resolver returns None (picked actor isn't a known GP
    diagram), the observer must not fire on_pick."""
    fake_picker = MagicMock()
    fake_picker.GetCellId.return_value = 0
    fake_picker.GetPickPosition.return_value = (0.0, 0.0, 0.0)
    fake_picker.GetActor.return_value = MagicMock()
    import vtk
    monkeypatch.setattr(vtk, "vtkCellPicker", lambda: fake_picker)

    plotter, iren = _stub_plotter()
    seen = []
    from apeGmsh.viewers.core.results_pick import install_results_pick
    ctrl = install_results_pick(
        plotter,
        on_pick=seen.append,
        scene=_stub_scene(),
        gp_resolver=lambda _a, _c: None,
    )
    handlers = _capture_handlers(iren)
    ctrl.set_mode("gp")

    _click(handlers, x=50, y=60)
    assert seen == []


# =====================================================================
# GP box-pick: drag in GP mode projects candidate GP centers + masks.
# =====================================================================

def test_gp_box_pick_dispatches_via_candidates(monkeypatch):
    import apeGmsh.viewers.core.results_pick as mod

    def fake_project(points, _renderer):
        return np.column_stack([
            np.arange(points.shape[0]) * 10.0,
            np.arange(points.shape[0]) * 10.0,
        ])
    monkeypatch.setattr(mod, "_project_points_to_display", fake_project)

    plotter, iren = _stub_plotter()
    box_seen: list = []

    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [3.0, 3.0, 0.0],
    ])
    element_ids = np.array([1001, 1001, 1002, 1003], dtype=np.int64)
    gp_indices = np.array([0, 1, 0, 0], dtype=np.int64)

    def candidates():
        return centers, element_ids, gp_indices

    from apeGmsh.viewers.core.results_pick import install_results_pick
    ctrl = install_results_pick(
        plotter,
        on_pick=lambda r: None,
        on_box_pick=box_seen.append,
        scene=_stub_scene(),
        gp_candidates=candidates,
    )
    handlers = _capture_handlers(iren)
    ctrl.set_mode("gp")

    # Drag from (0,0) to (25,25) — projection puts centers at
    # (0,0), (10,10), (20,20), (30,30); first three fall inside.
    handlers["LeftButtonPressEvent"](
        _stub_caller(x=0, y=0), "LeftButtonPressEvent",
    )
    handlers["MouseMoveEvent"](
        _stub_caller(x=25, y=25), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=25, y=25), "LeftButtonReleaseEvent",
    )

    assert len(box_seen) == 1
    box = box_seen[0]
    assert box.kind == "gp"
    np.testing.assert_array_equal(box.ids, [1001, 1001, 1002])
    np.testing.assert_array_equal(box.gp_indices, [0, 1, 0])


def test_gp_box_pick_with_no_candidates_returns_empty(monkeypatch):
    """When ``gp_candidates`` returns an empty result (no GP markers
    on screen), the box-pick path emits an empty BoxPickResult rather
    than firing the callback with a stale state."""
    plotter, iren = _stub_plotter()
    box_seen: list = []

    def candidates():
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    from apeGmsh.viewers.core.results_pick import install_results_pick
    ctrl = install_results_pick(
        plotter,
        on_pick=lambda r: None,
        on_box_pick=box_seen.append,
        scene=_stub_scene(),
        gp_candidates=candidates,
    )
    handlers = _capture_handlers(iren)
    ctrl.set_mode("gp")

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=0, y=0), "LeftButtonPressEvent",
    )
    handlers["MouseMoveEvent"](
        _stub_caller(x=25, y=25), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=25, y=25), "LeftButtonReleaseEvent",
    )

    assert len(box_seen) == 1
    box = box_seen[0]
    assert box.kind == "gp"
    assert box.ids.size == 0
    assert box.gp_indices.size == 0


def test_gp_box_pick_without_candidates_skips(monkeypatch):
    """Without ``gp_candidates`` wired, drag in GP mode is a silent
    no-op (no callback). Phase 2c default for GP."""
    plotter, iren = _stub_plotter()
    box_seen: list = []

    from apeGmsh.viewers.core.results_pick import install_results_pick
    ctrl = install_results_pick(
        plotter,
        on_pick=lambda r: None,
        on_box_pick=box_seen.append,
        scene=_stub_scene(),
        # gp_candidates=None
    )
    handlers = _capture_handlers(iren)
    ctrl.set_mode("gp")

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=0, y=0), "LeftButtonPressEvent",
    )
    handlers["MouseMoveEvent"](
        _stub_caller(x=25, y=25), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=25, y=25), "LeftButtonReleaseEvent",
    )

    assert box_seen == []


def test_gp_mode_without_resolver_skips_pick(monkeypatch):
    """When ``gp_resolver`` is None and the user is in GP mode, picks
    silently no-op rather than crashing."""
    fake_picker = MagicMock()
    fake_picker.GetCellId.return_value = 0
    fake_picker.GetPickPosition.return_value = (0.0, 0.0, 0.0)
    fake_picker.GetActor.return_value = MagicMock()
    import vtk
    monkeypatch.setattr(vtk, "vtkCellPicker", lambda: fake_picker)

    plotter, iren = _stub_plotter()
    seen, ctrl = _install(plotter)    # gp_resolver omitted (None)
    handlers = _capture_handlers(iren)
    ctrl.set_mode("gp")

    _click(handlers, x=50, y=60)
    assert seen == []


def test_box_pick_disabled_when_callback_is_none(monkeypatch):
    """Without on_box_pick, drag should be silently absorbed (no
    rubber-band rectangle, no callback). Phase 2a/2b behavior."""
    _patch_picker(monkeypatch, cell_id=42, world=(0.0, 0.0, 0.0))
    plotter, iren = _stub_plotter()
    scene = _stub_scene()
    seen, _ = _install(plotter, scene=scene)    # on_box_pick defaults to None
    handlers = _capture_handlers(iren)

    handlers["LeftButtonPressEvent"](
        _stub_caller(x=0, y=0), "LeftButtonPressEvent",
    )
    handlers["MouseMoveEvent"](
        _stub_caller(x=20, y=20), "MouseMoveEvent",
    )
    handlers["LeftButtonReleaseEvent"](
        _stub_caller(x=20, y=20), "LeftButtonReleaseEvent",
    )
    assert seen == []    # No click pick (was a drag).
    # No on_box_pick → silent no-op. The plotter's add_actor2D
    # mock should not have been called for a rubber-band actor
    # because we never called _ensure_rubberband.
    assert plotter.renderer.AddActor2D.call_count == 0
