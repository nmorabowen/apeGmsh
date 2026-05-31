"""ADR 0044 R-D.2 — ``PyVistaPickBackend`` (the shared pick backend).

Drives the stateless core (``resolve_pick`` / ``project_points`` /
``frustum_planes``) and the desktop gesture machine (``install`` →
press/move/release → callbacks, ``uninstall``) with fake interactor /
picker / renderer objects. No GPU, no live VTK context — the headless
testability ADR 0044 R-D.2 set out to unlock (pick logic previously
needed a real Qt event loop).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.backends._pyvista_pick import PyVistaPickBackend
from apeGmsh.viewers.scene_ir import (
    BoxGesture,
    PickBackend,
    PickHit,
    PickModifiers,
    PickRequest,
)


# ── Fakes ───────────────────────────────────────────────────────────

class _FakeCommand:
    def __init__(self) -> None:
        self.aborted = False

    def SetAbortFlag(self, v) -> None:
        self.aborted = bool(v)


class _FakeIren:
    """Stands in for the VTK interactor *and* the event ``caller``."""

    def __init__(self) -> None:
        self._by_event: dict = {}
        self._cmds: dict = {}
        self._next = 1
        self.removed: list = []
        self.event_pos = (0, 0)
        self.shift = False
        self.ctrl = False
        self.alt = False

    def AddObserver(self, event, cb, _pri):
        tag = self._next
        self._next += 1
        self._by_event.setdefault(event, []).append(cb)
        self._cmds[tag] = _FakeCommand()
        return tag

    def RemoveObserver(self, tag) -> None:
        self.removed.append(tag)

    def GetCommand(self, tag):
        return self._cmds.get(tag)

    def GetEventPosition(self):
        return self.event_pos

    def GetShiftKey(self):
        return self.shift

    def GetControlKey(self):
        return self.ctrl

    def GetAltKey(self):
        return self.alt

    def fire(self, event, pos=None) -> None:
        if pos is not None:
            self.event_pos = pos
        for cb in self._by_event.get(event, []):
            cb(self, None)


class _FakeRenderer:
    """Loop-fallback projection renderer (no GetActiveCamera)."""

    def __init__(self) -> None:
        self._wp = (0.0, 0.0, 0.0)

    def SetWorldPoint(self, x, y, z, _w) -> None:
        self._wp = (x, y, z)

    def WorldToDisplay(self) -> None:
        pass

    def GetDisplayPoint(self):
        # Trivial "projection": display = (x*2, y*2) so we can assert.
        return (self._wp[0] * 2.0, self._wp[1] * 2.0, 0.0)


class _FakePlotter:
    def __init__(self, iren, renderer) -> None:
        self.iren = type("I", (), {"interactor": iren})()
        self.renderer = renderer

    def render(self) -> None:
        pass


class _FakePicker:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def Pick(self, x, y, z, _renderer) -> None:
        self.cfg["picked"] = (x, y)

    def GetViewProp(self):
        return self.cfg["prop"]

    def GetCellId(self):
        return self.cfg["cell"]

    def GetPickPosition(self):
        return self.cfg["world"]


def _backend(monkeypatch, cfg, *, renderer=None):
    iren = _FakeIren()
    plotter = _FakePlotter(iren, renderer or _FakeRenderer())
    b = PyVistaPickBackend(plotter)
    monkeypatch.setattr(b, "_new_picker", lambda: _FakePicker(cfg))
    monkeypatch.setattr(b, "_update_rubberband", lambda *a, **k: None)
    return b, iren


# ── Protocol conformance ────────────────────────────────────────────

def test_satisfies_pick_backend_protocol() -> None:
    assert isinstance(PyVistaPickBackend(object()), PickBackend)


# ── resolve_pick ────────────────────────────────────────────────────

def test_resolve_pick_returns_hit(monkeypatch) -> None:
    prop = object()
    cfg = {"prop": prop, "cell": 7, "world": (1.0, 2.0, 3.0)}
    b, _ = _backend(monkeypatch, cfg)
    hit = b.resolve_pick(PickRequest(5, 6))
    assert isinstance(hit, PickHit)
    assert hit.prop_id == id(prop)
    assert hit.cell_id == 7
    assert hit.world == (1.0, 2.0, 3.0)


def test_resolve_pick_miss_returns_none(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    b, _ = _backend(monkeypatch, cfg)
    assert b.resolve_pick(PickRequest(5, 6)) is None


# ── project_points ──────────────────────────────────────────────────

def test_project_points_loop_fallback(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    b, _ = _backend(monkeypatch, cfg)
    pts = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
    out = b.project_points(pts)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[2.0, 4.0], [6.0, 8.0]])


# ── frustum_planes ──────────────────────────────────────────────────

def test_frustum_planes_forced_2d_returns_none(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    b, _ = _backend(monkeypatch, cfg)
    monkeypatch.setenv("APEGMSH_BOX_2D", "1")
    assert b.frustum_planes((0, 0, 10, 10)) is None


def test_frustum_planes_none_on_unprojectable_renderer(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    # _FakeRenderer has no DisplayToWorld → _unproject raises → None.
    b, _ = _backend(monkeypatch, cfg)
    monkeypatch.delenv("APEGMSH_BOX_2D", raising=False)
    assert b.frustum_planes((0, 0, 10, 10)) is None


# ── install: click ──────────────────────────────────────────────────

def test_click_fires_on_pick(monkeypatch) -> None:
    prop = object()
    cfg = {"prop": prop, "cell": 4, "world": (9.0, 8.0, 7.0)}
    b, iren = _backend(monkeypatch, cfg)
    picks: list = []
    b.install(on_pick=lambda hit, mods: picks.append((hit, mods)))
    iren.fire("LeftButtonPressEvent", pos=(10, 20))
    iren.fire("LeftButtonReleaseEvent", pos=(10, 20))
    assert len(picks) == 1
    hit, mods = picks[0]
    assert hit.prop_id == id(prop) and hit.cell_id == 4
    assert mods == PickModifiers()


def test_click_captures_modifiers(monkeypatch) -> None:
    cfg = {"prop": object(), "cell": 0, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    picks: list = []
    b.install(on_pick=lambda hit, mods: picks.append(mods))
    iren.ctrl = True
    iren.alt = True
    iren.fire("LeftButtonPressEvent", pos=(1, 1))
    iren.ctrl = iren.alt = False  # release-time modifiers must not matter
    iren.fire("LeftButtonReleaseEvent", pos=(1, 1))
    assert picks == [PickModifiers(ctrl=True, alt=True)]


def test_shift_press_suppresses_pick(monkeypatch) -> None:
    cfg = {"prop": object(), "cell": 0, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    picks: list = []
    b.install(on_pick=lambda hit, mods: picks.append(hit))
    iren.shift = True
    iren.fire("LeftButtonPressEvent", pos=(5, 5))
    iren.shift = False
    iren.fire("LeftButtonReleaseEvent", pos=(5, 5))
    assert picks == []  # navigation owns Shift+LMB


# ── install: box drag ───────────────────────────────────────────────

def test_drag_fires_on_box_window(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    boxes: list = []
    b.install(on_pick=lambda *a: None, on_box=boxes.append)
    iren.fire("LeftButtonPressEvent", pos=(10, 20))
    iren.fire("MouseMoveEvent", pos=(60, 80))   # dist > threshold → drag
    iren.fire("LeftButtonReleaseEvent", pos=(60, 80))
    assert len(boxes) == 1
    g = boxes[0]
    assert isinstance(g, BoxGesture)
    assert g.box == (10, 20, 60, 80)
    assert g.crossing is False  # left→right


def test_drag_right_to_left_is_crossing(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    boxes: list = []
    b.install(on_pick=lambda *a: None, on_box=boxes.append)
    iren.fire("LeftButtonPressEvent", pos=(60, 20))
    iren.fire("MouseMoveEvent", pos=(10, 80))
    iren.fire("LeftButtonReleaseEvent", pos=(10, 80))
    assert boxes[0].crossing is True


def test_tiny_drag_is_a_click_not_box(monkeypatch) -> None:
    cfg = {"prop": object(), "cell": 0, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    picks: list = []
    boxes: list = []
    b.install(on_pick=lambda hit, mods: picks.append(hit), on_box=boxes.append)
    iren.fire("LeftButtonPressEvent", pos=(10, 20))
    iren.fire("MouseMoveEvent", pos=(12, 21))   # within 8px threshold
    iren.fire("LeftButtonReleaseEvent", pos=(12, 21))
    assert len(picks) == 1 and boxes == []


# ── install: hover ──────────────────────────────────────────────────

def test_hover_fires_after_throttle(monkeypatch) -> None:
    prop = object()
    cfg = {"prop": prop, "cell": 1, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    hovers: list = []
    b.install(on_pick=lambda *a: None, on_hover=hovers.append)
    # No press → idle move. Throttle is 1-in-3.
    iren.fire("MouseMoveEvent", pos=(1, 1))
    iren.fire("MouseMoveEvent", pos=(2, 2))
    assert hovers == []
    iren.fire("MouseMoveEvent", pos=(3, 3))
    assert len(hovers) == 1 and hovers[0].prop_id == id(prop)


# ── uninstall ───────────────────────────────────────────────────────

def test_uninstall_removes_observers(monkeypatch) -> None:
    cfg = {"prop": None, "cell": -1, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    b.install(on_pick=lambda *a: None)
    b.uninstall()
    assert len(iren.removed) == 3
    # Idempotent.
    b.uninstall()


def test_pick_after_uninstall_is_silent(monkeypatch) -> None:
    cfg = {"prop": object(), "cell": 0, "world": (0.0, 0.0, 0.0)}
    b, iren = _backend(monkeypatch, cfg)
    picks: list = []
    b.install(on_pick=lambda hit, mods: picks.append(hit))
    b.uninstall()
    # Observers are gone from a real iren; our fake still holds the cb,
    # but the press state was reset, so a release alone fires nothing.
    iren.fire("LeftButtonReleaseEvent", pos=(1, 1))
    assert picks == []
