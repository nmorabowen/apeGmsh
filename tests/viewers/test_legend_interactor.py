"""LegendInteractor gestures (ADR 0081 L2).

The scale drew a drag handle from day one and never moved: the widget
that owned it observed the interactor at VTK priority 0.5 while the pick
engine aborts every plain LMB press at 10. These cover the replacement —
gestures in the event stream the application already owns.

The load-bearing test is ``test_a_miss_does_not_abort``: the interactor
sits *above* navigation and picking, so if it claimed events it did not
need, it would break camera and selection everywhere outside a legend.
"""
from __future__ import annotations

import pytest

from apeGmsh.viewers.core._legend import LegendController, MAX_FONT_SCALE
from apeGmsh.viewers.core._legend_interactor import (
    HANDLE_PX,
    MODE_MOVE,
    MODE_RESIZE,
    LegendInteractor,
    install_legend_interactor,
)
from apeGmsh.viewers.scene_ir import LutSpec


# =====================================================================
# Fakes — a scriptable VTK interactor
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
        self.repeat = 0

    # -- VTK surface --
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
        return self.repeat

    # -- test surface --
    def send(self, event, pos=None, *, shift=False, repeat=0):
        """Fire ``event`` at ``pos``; returns True if someone aborted."""
        if pos is not None:
            self.pos = pos
        self.shift, self.repeat = shift, repeat
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


class _Backend:
    def __init__(self, viewport=(1000, 800)) -> None:
        self.viewport = viewport
        self.bars: dict = {}
        self.iren = _Iren()
        self.render_window = _Window()
        outer = self
        self.plotter = type(
            "P", (), {
                "iren": type("W", (), {"interactor": outer.iren})(),
                "render_window": outer.render_window,
            },
        )()

    def viewport_size(self):
        return self.viewport

    def add_scalar_bar(self, handle, spec):
        self.bars[spec.key] = spec

    def remove_scalar_bar(self, key):
        self.bars.pop(key, None)


class _Handle:
    def __init__(self, layer_id="L0") -> None:
        self.layer_id = layer_id


@pytest.fixture
def rig():
    backend = _Backend()
    legends = LegendController(backend)
    legends.register(
        ("", "stress_vm"), "L0", _Handle(), lut=LutSpec(vmin=-1.0, vmax=1.0),
    )
    interactor = install_legend_interactor(backend, legends)
    assert interactor is not None
    return backend, legends, interactor


def _entry(legends):
    return legends.entry(("", "stress_vm"))


def _px(legends, nx, ny):
    vw, vh = legends.viewport_px()
    return int(nx * vw), int(ny * vh)


def _centre(legends):
    e = _entry(legends)
    return _px(legends, e.anchor[0] + e.extent[0] / 2,
               e.anchor[1] + e.extent[1] / 2)


# =====================================================================
# L-PICK — the interactor claims only what it is over
# =====================================================================

def test_a_miss_does_not_abort(rig):
    """Outside a legend, camera and picking behave exactly as before.

    The interactor sits above navigation (11) and picking (10), so an
    over-eager abort here would break selection and orbit everywhere.
    """
    _backend, legends, _it = rig
    assert legends.viewport_px() == (1000, 800)
    assert _entry(legends).anchor[0] > 0.5      # docked right; miss on the left

    for event in ("LeftButtonPressEvent", "MouseMoveEvent",
                  "LeftButtonReleaseEvent", "RightButtonPressEvent"):
        assert _backend.iren.send(event, (20, 400)) is False, event


def test_a_hit_aborts(rig):
    backend, legends, _it = rig
    assert backend.iren.send("LeftButtonPressEvent", _centre(legends)) is True


def test_shift_lmb_over_a_legend_is_left_to_navigation(rig):
    """Shift+LMB is the orbit gesture; the legend never steals it."""
    backend, legends, _it = rig
    assert backend.iren.send(
        "LeftButtonPressEvent", _centre(legends), shift=True,
    ) is False


def test_installed_above_navigation_and_picking(rig):
    backend, _legends, _it = rig
    priorities = {p for _e, _cb, p in backend.iren.observers.values()}
    assert priorities == {12.0}


def test_uninstall_removes_every_observer(rig):
    backend, legends, it = rig
    over_a_legend = _centre(legends)
    it.uninstall()
    assert backend.iren.observers == {}
    # Nothing left to claim the event, legend or no legend.
    assert backend.iren.send("LeftButtonPressEvent", over_a_legend) is False


# =====================================================================
# Hit testing
# =====================================================================

def test_body_hit_is_a_move_and_the_corner_is_a_resize(rig):
    _backend, legends, it = rig
    e = _entry(legends)
    vw, vh = legends.viewport_px()

    body = _px(legends, e.anchor[0] + e.extent[0] / 2,
               e.anchor[1] + e.extent[1] / 2)
    assert it.hit(*body) == (e, MODE_MOVE)

    corner = (
        int((e.anchor[0] + e.extent[0]) * vw) - HANDLE_PX // 2,
        int((e.anchor[1] + e.extent[1]) * vh) - HANDLE_PX // 2,
    )
    assert it.hit(*corner) == (e, MODE_RESIZE)


def test_a_hidden_legend_is_not_hit(rig):
    _backend, legends, it = rig
    pos = _centre(legends)
    legends.set_visible(("", "stress_vm"), False)
    assert it.hit(*pos) == (None, None)


# =====================================================================
# Move
# =====================================================================

def test_drag_moves_the_legend_and_undocks_it(rig):
    backend, legends, _it = rig
    start = _centre(legends)
    before = _entry(legends).anchor

    backend.iren.send("LeftButtonPressEvent", start)
    backend.iren.send("MouseMoveEvent", (start[0] - 300, start[1] - 200))
    backend.iren.send("LeftButtonReleaseEvent")

    e = _entry(legends)
    assert e.anchor[0] < before[0] and e.anchor[1] < before[1]
    assert e.slot is None, "a hand-placed legend must leave the stack"


def test_a_drag_survives_a_later_recolour(rig):
    """The whole point of L2 — the gesture writes real state.

    Pre-0081 the widget's geometry was a copy nothing read back, so even
    a working drag was erased by the next colormap change.
    """
    backend, legends, _it = rig
    start = _centre(legends)
    backend.iren.send("LeftButtonPressEvent", start)
    backend.iren.send("MouseMoveEvent", (start[0] - 250, start[1] - 150))
    backend.iren.send("LeftButtonReleaseEvent")
    placed = _entry(legends).anchor

    legends.set_lut(("", "stress_vm"), LutSpec(name="turbo", vmin=-9, vmax=9))
    legends.set_fmt(("", "stress_vm"), "%.4f")
    assert _entry(legends).anchor == pytest.approx(placed)


def test_drop_near_the_home_edge_redocks(rig):
    backend, legends, _it = rig
    start = _centre(legends)
    backend.iren.send("LeftButtonPressEvent", start)
    backend.iren.send("MouseMoveEvent", (start[0] - 400, start[1]))
    backend.iren.send("LeftButtonReleaseEvent")
    assert _entry(legends).slot is None

    # Drag it back against the right-hand edge it docks to.
    mid = _centre(legends)
    backend.iren.send("LeftButtonPressEvent", mid)
    backend.iren.send("MouseMoveEvent", (990, mid[1]))
    backend.iren.send("LeftButtonReleaseEvent")
    assert _entry(legends).slot == 0


def test_double_click_redocks(rig):
    backend, legends, _it = rig
    legends.set_anchor(("", "stress_vm"), (0.3, 0.3))
    assert _entry(legends).slot is None

    backend.iren.send("LeftButtonPressEvent", _centre(legends), repeat=1)
    assert _entry(legends).slot == 0


def test_a_dragged_legend_stays_inside_the_viewport(rig):
    backend, legends, _it = rig
    start = _centre(legends)
    backend.iren.send("LeftButtonPressEvent", start)
    for target in ((-500, -500), (5000, 5000), (0, 5000)):
        backend.iren.send("MouseMoveEvent", target)
        e = _entry(legends)
        assert 0.0 <= e.anchor[0] <= 1.0 and 0.0 <= e.anchor[1] <= 1.0
        assert e.anchor[0] + e.extent[0] <= 1.0 + 1e-9
        assert e.anchor[1] + e.extent[1] <= 1.0 + 1e-9


# =====================================================================
# Resize
# =====================================================================

def test_corner_drag_outward_grows_the_legend(rig):
    backend, legends, _it = rig
    e = _entry(legends)
    vw, vh = legends.viewport_px()
    corner = (
        int((e.anchor[0] + e.extent[0]) * vw) - 2,
        int((e.anchor[1] + e.extent[1]) * vh) - 2,
    )
    before = e.extent

    backend.iren.send("LeftButtonPressEvent", corner)
    backend.iren.send("MouseMoveEvent", (corner[0] + 120, corner[1] + 120))
    backend.iren.send("LeftButtonReleaseEvent")

    e = _entry(legends)
    assert e.font_scale > 1.0
    assert e.extent[0] > before[0] and e.extent[1] > before[1]


def test_corner_drag_inward_shrinks_the_legend(rig):
    backend, legends, _it = rig
    legends.set_font_scale(("", "stress_vm"), 2.0)
    e = _entry(legends)
    vw, vh = legends.viewport_px()
    corner = (
        int((e.anchor[0] + e.extent[0]) * vw) - 2,
        int((e.anchor[1] + e.extent[1]) * vh) - 2,
    )

    backend.iren.send("LeftButtonPressEvent", corner)
    backend.iren.send("MouseMoveEvent", (corner[0] - 60, corner[1] - 60))
    assert _entry(legends).font_scale < 2.0


def test_resize_cannot_drive_the_legend_past_its_bounds(rig):
    """A wild drag clamps instead of inverting the box."""
    backend, legends, _it = rig
    e = _entry(legends)
    vw, vh = legends.viewport_px()
    corner = (
        int((e.anchor[0] + e.extent[0]) * vw) - 2,
        int((e.anchor[1] + e.extent[1]) * vh) - 2,
    )
    backend.iren.send("LeftButtonPressEvent", corner)
    for target in ((5000, 5000), (int(e.anchor[0] * vw), int(e.anchor[1] * vh))):
        backend.iren.send("MouseMoveEvent", target)
        entry = _entry(legends)
        assert 0 < entry.extent[0] <= 1.0 and 0 < entry.extent[1] <= 1.0
        assert entry.font_scale <= MAX_FONT_SCALE


# =====================================================================
# Cursor + context menu
# =====================================================================

def test_hover_sets_and_clears_the_cursor(rig):
    backend, legends, _it = rig
    backend.iren.send("MouseMoveEvent", _centre(legends))
    over = backend.render_window.cursor
    assert over is not None

    backend.iren.send("MouseMoveEvent", (20, 400))
    assert backend.render_window.cursor != over


def test_right_click_over_a_legend_offers_the_menu():
    backend = _Backend()
    legends = LegendController(backend)
    legends.register(("", "vm"), "L0", _Handle(), lut=LutSpec())
    seen: list = []
    install_legend_interactor(
        backend, legends,
        on_context_menu=lambda entry, pos: seen.append((entry.key, pos)),
    )
    e = legends.entry(("", "vm"))
    vw, vh = legends.viewport_px()
    pos = (int((e.anchor[0] + e.extent[0] / 2) * vw),
           int((e.anchor[1] + e.extent[1] / 2) * vh))

    assert backend.iren.send("RightButtonPressEvent", pos) is True
    assert len(seen) == 1
    assert seen[0][0] == ("", "vm")
    # Reported top-left-origin, as Qt menus expect.
    assert seen[0][1][1] == vh - pos[1]


def test_right_click_off_a_legend_stays_navigation_pan():
    backend = _Backend()
    legends = LegendController(backend)
    legends.register(("", "vm"), "L0", _Handle(), lut=LutSpec())
    seen: list = []
    install_legend_interactor(
        backend, legends, on_context_menu=lambda e, p: seen.append(e),
    )
    assert backend.iren.send("RightButtonPressEvent", (20, 400)) is False
    assert seen == []


def test_install_returns_none_without_an_interactor():
    class _Headless:
        def viewport_size(self):
            return (800, 600)

    assert install_legend_interactor(_Headless(), None) is None


# =====================================================================
# Drag hygiene — findings from L2's own adversarial pass
# =====================================================================

def test_a_drag_does_not_rebuild_the_bar_actor_per_frame(rig):
    """A move emits one event per frame; rebuilding the actor flickers.

    Measured before the fix: 40 mouse-move events destroyed and
    re-created the VTK bar actor 40 times.
    """
    backend, legends, _it = rig
    rebuilt: list = []
    moved: list = []
    backend.add_scalar_bar = lambda h, s: (
        rebuilt.append(s.key), backend.bars.__setitem__(s.key, s),
    )
    backend.move_scalar_bar = lambda k, s: (
        moved.append(k), backend.bars.__setitem__(k, s), True,
    )[-1]

    start = _centre(legends)
    backend.iren.send("LeftButtonPressEvent", start)
    for i in range(1, 41):      # from 1: a zero-delta move is a no-op
        backend.iren.send("MouseMoveEvent", (start[0] - i * 4, start[1] - i * 2))
    backend.iren.send("LeftButtonReleaseEvent")

    assert rebuilt == [], "geometry-only changes must not rebuild the actor"
    assert len(moved) == 40


def test_a_lut_change_still_rebuilds_the_bar(rig):
    """The in-place fast path must not swallow a real content change."""
    backend, legends, _it = rig
    rebuilt: list = []
    backend.add_scalar_bar = lambda h, s: (
        rebuilt.append(s.key), backend.bars.__setitem__(s.key, s),
    )
    backend.move_scalar_bar = lambda k, s: True

    legends.set_lut(("", "stress_vm"), LutSpec(name="turbo", vmin=-9, vmax=9))
    legends.set_vertical(("", "stress_vm"), False)
    legends.set_fmt(("", "stress_vm"), "%.4f")
    assert len(rebuilt) == 3


def test_leaving_the_window_ends_a_drag(rig):
    """A release delivered outside the window never reaches us.

    Without this, the legend follows the cursor forever afterwards.
    """
    backend, legends, _it = rig
    start = _centre(legends)
    backend.iren.send("LeftButtonPressEvent", start)
    backend.iren.send("MouseMoveEvent", (start[0] - 100, start[1]))
    dragged_to = _entry(legends).anchor

    backend.iren.send("LeaveEvent")
    backend.iren.send("MouseMoveEvent", (50, 50))
    assert _entry(legends).anchor == pytest.approx(dragged_to)


def test_hover_sets_the_cursor_only_when_it_changes(rig):
    backend, legends, it = rig
    calls: list = []
    backend.render_window.SetCurrentCursor = calls.append

    over = _centre(legends)
    for _ in range(5):
        backend.iren.send("MouseMoveEvent", over)
    assert len(calls) == 1, "re-asserted the same cursor every frame"

    for _ in range(5):
        backend.iren.send("MouseMoveEvent", (20, 400))
    assert len(calls) == 2
