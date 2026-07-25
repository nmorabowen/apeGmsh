"""LegendController layout — the picture, not the API (ADR 0081 Part 6).

The suite these replace asserted that we called VTK ("a bar is
registered under title X", "GetLabelFormat returns the fmt") and stayed
green through every defect users actually reported. These assert what
lands on screen: the box measured in pixels, the text fitting inside it,
and legends not sitting on top of each other.

* **L-GEOM** — the rendered actor's box matches the declared extent.
* **L-FIT** — the content fits the box at every allowed size, and the
  box stays inside the viewport for both orientations.
* **L-NOOVERLAP** — N docked legends never intersect.
"""
from __future__ import annotations

import weakref

import pytest

from apeGmsh.viewers.core._legend import (
    CHAR_ADVANCE,
    MAX_FONT_SCALE,
    MIN_FONT_SCALE,
    LegendController,
    LegendEntry,
    box_px,
    measure,
    text_width_px,
    tick_labels,
)
from apeGmsh.viewers.scene_ir import LutSpec


# =====================================================================
# Helpers
# =====================================================================

class _Handle:
    def __init__(self, layer_id: str) -> None:
        self.layer_id = layer_id


class _Sink:
    """Minimal backend: records specs, reports a fixed viewport."""

    def __init__(self, viewport=(1280, 800)) -> None:
        self.viewport = viewport
        self.bars: dict = {}

    def viewport_size(self):
        return self.viewport

    def add_scalar_bar(self, handle, spec):
        self.bars[spec.key] = spec

    def remove_scalar_bar(self, bar_key):
        self.bars.pop(bar_key, None)


def _controller(viewport=(1280, 800)):
    sink = _Sink(viewport)
    return LegendController(sink), sink


def _register(ctl, component, *, geometry="", vertical=False,
              vmin=-0.5, vmax=0.5, fmt="%.3g"):
    key = (geometry, component)
    return ctl.register(
        key, f"layer::{geometry}::{component}", _Handle(f"L{component}"),
        lut=LutSpec(vmin=vmin, vmax=vmax), fmt=fmt, vertical=vertical,
    )


def _rects(ctl):
    """(x0, y0, x1, y1) for every visible legend, in viewport units."""
    out = []
    for e in ctl.layout():
        x, y = e.anchor
        w, h = e.extent
        out.append((x, y, x + w, y + h))
    return out


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# =====================================================================
# L-GEOM — the declared extent is what renders
# =====================================================================

@pytest.mark.parametrize("vertical", [True, False])
def test_rendered_box_matches_declared_extent(pv_backend, vertical):
    """The bar actor's on-screen box is the spec's, to the pixel.

    The pre-0081 path passed a layout and then let an interactive widget
    re-derive its own geometry, so what rendered was not what was asked
    for. The backend may not reinterpret the spec.
    """
    import pyvista as pv

    mesh = pv.Sphere()
    mesh["v"] = mesh.points[:, 2]
    actor = pv_backend.plotter.add_mesh(mesh, scalars="v", show_scalar_bar=False)

    ctl = LegendController(pv_backend)
    entry = ctl.register(
        ("", "displacement_z"), "L0", _PvHandle(actor),
        lut=LutSpec(vmin=-0.5, vmax=0.5), vertical=vertical,
    )
    spec = ctl.specs()[0]

    bar = pv_backend.plotter.scalar_bars["displacement_z"]
    assert bar.GetWidth() == pytest.approx(spec.extent[0], abs=1e-6)
    assert bar.GetHeight() == pytest.approx(spec.extent[1], abs=1e-6)
    assert bar.GetPosition()[0] == pytest.approx(spec.anchor[0], abs=1e-6)
    assert bar.GetPosition()[1] == pytest.approx(spec.anchor[1], abs=1e-6)
    # The bar sits inside the legend footprint the layout reserved; a
    # horizontal legend gives part of it back for the title band.
    assert spec.extent[1] <= entry.extent[1] + 1e-9
    assert (spec.extent[1] < entry.extent[1]) is (not vertical)


def test_horizontal_title_is_drawn_outside_the_bar(pv_backend):
    """A horizontal legend's title gets its own band above the bar.

    ``vtkScalarBarActor`` centres a horizontal title inside the
    tick-label band, landing it on top of the middle label at every box
    size and in both font-sizing modes. So the controller reserves a
    band and the backend draws the title into it; VTK must not draw one
    as well.
    """
    import pyvista as pv

    mesh = pv.Sphere()
    mesh["v"] = mesh.points[:, 2]
    actor = pv_backend.plotter.add_mesh(mesh, scalars="v", show_scalar_bar=False)

    ctl = LegendController(pv_backend)
    ctl.register(
        ("", "displacement_z"), "L0", _PvHandle(actor),
        lut=LutSpec(vmin=-0.5, vmax=0.5), vertical=False,
    )
    spec = ctl.specs()[0]
    bar = pv_backend.plotter.scalar_bars["displacement_z"]

    assert spec.title_anchor is not None
    assert bar.GetTitle() == "", "VTK would overlap its own title"
    # The title anchor is above the bar box and horizontally centred.
    assert spec.title_anchor[1] >= spec.anchor[1] + spec.extent[1] - 1e-9
    assert spec.title_anchor[0] == pytest.approx(
        spec.anchor[0] + spec.extent[0] / 2, abs=1e-6
    )


def test_vertical_legend_lets_vtk_draw_its_own_title(pv_backend):
    """VTK's vertical layout is the one it gets right — leave it alone."""
    import pyvista as pv

    mesh = pv.Sphere()
    mesh["v"] = mesh.points[:, 2]
    actor = pv_backend.plotter.add_mesh(mesh, scalars="v", show_scalar_bar=False)

    ctl = LegendController(pv_backend)
    ctl.register(
        ("", "displacement_z"), "L0", _PvHandle(actor),
        lut=LutSpec(vmin=-0.5, vmax=0.5), vertical=True,
    )
    spec = ctl.specs()[0]
    bar = pv_backend.plotter.scalar_bars["displacement_z"]
    assert spec.title_anchor is None
    assert bar.GetTitle() == "displacement_z"


def test_legends_default_to_vertical():
    """Vertical is VTK's correct layout, and it spares the model view."""
    ctl, _ = _controller()
    entry = ctl.register(("", "sig"), "L0", _Handle("L0"), lut=LutSpec())
    assert entry.vertical is True


class _PvHandle:
    """A LayerHandle-shaped wrapper around a raw pyvista actor."""

    def __init__(self, actor) -> None:
        self.actor = actor
        self.layer_id = "L0"


def test_no_interactive_widget_is_created(pv_backend):
    """No ``vtkScalarBarWidget`` — it could never receive a click.

    It observes the interactor at priority 0.5 while the pick engine
    aborts every plain LMB press at 10, so the handle it drew was
    unreachable by construction (ADR 0081 D1).
    """
    import pyvista as pv

    mesh = pv.Sphere()
    mesh["v"] = mesh.points[:, 2]
    actor = pv_backend.plotter.add_mesh(mesh, scalars="v", show_scalar_bar=False)

    ctl = LegendController(pv_backend)
    ctl.register(("", "sig"), "L0", _PvHandle(actor), lut=LutSpec())
    ctl.layout()

    widgets = getattr(pv_backend.plotter.scalar_bars, "_scalar_bar_widgets", {})
    assert widgets == {}


# =====================================================================
# L-FIT — the content fits, at every size, in both orientations
# =====================================================================

def test_advance_estimate_brackets_measured_text():
    """The width model is conservative but not absurd.

    Non-circular on purpose: the estimate is checked against a real VTK
    text extent, not against itself. It must never under-estimate (a
    narrow box clips labels) and must not be wildly padded either.
    """
    import vtk

    renderer = vtk.vtkTextRenderer.GetInstance()
    for text, pt in [("displacement_z", 14), ("-0.25", 12), ("1.2345e-03", 16)]:
        prop = vtk.vtkTextProperty()
        prop.SetFontSize(int(pt))
        prop.SetFontFamilyToArial()
        bbox = [0, 0, 0, 0]
        assert renderer.GetBoundingBox(prop, text, bbox, 72)
        measured = bbox[1] - bbox[0]

        estimate = text_width_px(text, pt)
        assert estimate >= measured * 0.98, (
            f"{text!r} at {pt}pt: estimate {estimate:.1f}px under-measures "
            f"{measured}px — labels would clip"
        )
        assert estimate <= measured * 1.9, (
            f"{text!r} at {pt}pt: estimate {estimate:.1f}px is "
            f"{estimate / max(measured, 1):.2f}× the measured {measured}px — "
            f"box needlessly bloated"
        )


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("scale", [MIN_FONT_SCALE, 0.75, 1.0, 2.0, MAX_FONT_SCALE])
def test_box_fits_its_own_text(vertical, scale):
    """The box always has room for the title and every tick label.

    This is D3's regression test: the old ``size`` multiplier shrank the
    box while the fonts stayed at 18 pt, so 0.35× produced a legend
    whose title overlapped its labels.
    """
    entry = LegendEntry(
        key=("", "displacement_z"), component="displacement_z",
        lut=LutSpec(vmin=-1234.5, vmax=1234.5), vertical=vertical,
        font_scale=scale,
    )
    m = measure(entry, scale)
    w, h = box_px(entry, m)

    assert w >= m.title_w, "box narrower than its title"
    assert w >= m.label_w, "box narrower than a tick label"
    if vertical:
        assert h >= entry.n_labels * m.line_h, "labels would collide vertically"
    else:
        assert w >= entry.n_labels * m.label_w, "labels would collide horizontally"
        assert h >= m.line_h + m.title_pt, "no room for title above labels"


def test_wider_format_widens_the_box():
    """A wider fmt grows the legend instead of overflowing it."""
    ctl, _ = _controller()
    entry = _register(ctl, "sig", fmt="%.3g")
    narrow = entry.extent
    ctl.set_fmt(("", "sig"), "%.8f")
    ctl.layout()
    assert entry.extent[0] > narrow[0]


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("scale", [MIN_FONT_SCALE, 1.0, MAX_FONT_SCALE])
def test_legend_stays_inside_the_viewport(vertical, scale):
    """Box + margins never leave the viewport.

    D5: the stock vertical anchor at x=0.9 pushed a centred title off
    the right edge of the window.
    """
    ctl, _ = _controller()
    ctl.default_font_scale = scale
    _register(ctl, "displacement_magnitude", vertical=vertical)
    for x0, y0, x1, y1 in _rects(ctl):
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0


def test_small_viewport_still_produces_a_contained_box():
    """A legend too big for a tiny window is clamped, not overflowed."""
    ctl, _ = _controller(viewport=(320, 240))
    ctl.default_font_scale = MAX_FONT_SCALE
    _register(ctl, "a_very_long_component_name_indeed")
    for x0, y0, x1, y1 in _rects(ctl):
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0


# =====================================================================
# L-NOOVERLAP — N legends coexist
# =====================================================================

@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_docked_legends_do_not_overlap(vertical, n):
    """D4: three diagrams used to stack three bars at (0.35, 0.05)."""
    ctl, _ = _controller()
    for i in range(n):
        _register(ctl, f"component_{i}", vertical=vertical)
    rects = _rects(ctl)
    assert len(rects) == n
    for i in range(n):
        for j in range(i + 1, n):
            assert not _overlap(rects[i], rects[j]), (
                f"legend {i} overlaps {j}: {rects[i]} vs {rects[j]}"
            )


def test_mixed_orientations_do_not_overlap():
    """Horizontal and vertical legends dock to different edges."""
    ctl, _ = _controller()
    _register(ctl, "sig_h", vertical=False)
    _register(ctl, "sig_v", vertical=True)
    _register(ctl, "sig_h2", vertical=False)
    rects = _rects(ctl)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlap(rects[i], rects[j])


def test_two_diagrams_on_one_component_share_one_legend():
    """One array, one scale — not two bars fighting for the same spot."""
    ctl, sink = _controller()
    key = ("", "stress_vm")
    ctl.register(key, "layerA", _Handle("layerA"), lut=LutSpec())
    ctl.register(key, "layerB", _Handle("layerB"), lut=LutSpec())
    assert len(ctl.entries()) == 1
    assert len(sink.bars) == 1
    # The legend survives the first diagram's detach.
    ctl.unregister("layerA")
    assert len(ctl.entries()) == 1
    ctl.unregister("layerB")
    assert ctl.entries() == []
    assert sink.bars == {}


def test_concurrent_geometries_keep_separate_legends():
    """Same component on two geometries → two keys, two legends.

    Keying on ``(geometry, component)`` is what lets add and remove
    agree; the old on-demand title prefix could change under the
    diagram between the two.
    """
    ctl, sink = _controller()
    _register(ctl, "stress_vm", geometry="Geometry 1")
    _register(ctl, "stress_vm", geometry="Geometry 2")
    assert len(ctl.entries()) == 2
    assert len(sink.bars) == 2
    titles = {s.title for s in sink.bars.values()}
    assert titles == {"Geometry 1 — stress_vm", "Geometry 2 — stress_vm"}


def test_single_geometry_title_collapses_to_the_component():
    """The geometry prefix only appears once it disambiguates."""
    ctl, sink = _controller()
    _register(ctl, "stress_vm", geometry="Geometry 1")
    assert next(iter(sink.bars.values())).title == "stress_vm"


# =====================================================================
# Slots, hiding, and hand placement
# =====================================================================

def test_hidden_legend_leaves_no_bar_and_frees_its_slot():
    ctl, sink = _controller()
    _register(ctl, "a")
    _register(ctl, "b")
    stacked_y = ctl.entry(("", "b")).anchor[1]

    ctl.set_visible(("", "a"), False)
    assert set(sink.bars) == {"b"}
    assert len(_rects(ctl)) == 1
    # 'b' moves down into the slot 'a' vacated.
    assert ctl.entry(("", "b")).anchor[1] < stacked_y


def test_hand_placed_anchor_is_honoured_and_survives_a_relayout():
    """An undocked legend keeps its anchor across recomputations.

    The seed of L-STICKY (ADR 0081 L1): the controller must never
    re-derive a user-placed position from the style.
    """
    ctl, _ = _controller()
    _register(ctl, "a")
    _register(ctl, "b")
    ctl.set_anchor(("", "a"), (0.62, 0.71))
    ctl.layout()
    assert ctl.entry(("", "a")).anchor == pytest.approx((0.62, 0.71))
    ctl.set_fmt(("", "b"), "%.5f")
    ctl.layout()
    assert ctl.entry(("", "a")).anchor == pytest.approx((0.62, 0.71))


def test_redock_returns_a_hand_placed_legend_to_the_stack():
    ctl, _ = _controller()
    _register(ctl, "a")
    ctl.set_anchor(("", "a"), (0.8, 0.8))
    assert ctl.entry(("", "a")).slot is None
    ctl.redock(("", "a"))
    assert ctl.entry(("", "a")).slot == 0
    ctl.layout()
    assert ctl.entry(("", "a")).anchor[1] < 0.2   # back on the bottom edge


def test_tick_labels_follow_the_lut_range_and_format():
    entry = LegendEntry(
        key=("", "v"), component="v", lut=LutSpec(vmin=0.0, vmax=100.0),
        fmt="%.1f", n_labels=5,
    )
    assert tick_labels(entry) == ["0.0", "25.0", "50.0", "75.0", "100.0"]


def test_bad_format_string_degrades_instead_of_raising():
    """The user types into the fmt field live; a typo must not crash."""
    entry = LegendEntry(
        key=("", "v"), component="v", lut=LutSpec(vmin=0.0, vmax=1.0),
        fmt="%.3q",
    )
    assert len(tick_labels(entry)) == 5


def test_font_scale_is_clamped_to_the_documented_range():
    ctl, _ = _controller()
    _register(ctl, "a")
    ctl.set_font_scale(("", "a"), 99.0)
    assert ctl.entry(("", "a")).font_scale == MAX_FONT_SCALE
    ctl.set_font_scale(("", "a"), 0.0)
    assert ctl.entry(("", "a")).font_scale == MIN_FONT_SCALE


def test_char_advance_is_documented_as_conservative():
    """Guard the constant the width model rests on."""
    assert 0.5 <= CHAR_ADVANCE <= 0.75


# =====================================================================
# Lifetime, aliasing and degenerate viewports
#
# Every case below was a live defect found by adversarially probing the
# first cut of this module.
# =====================================================================

def test_controller_does_not_keep_its_backend_alive():
    """``controller_for`` must not leak the viewport it is keyed by.

    A ``WeakKeyDictionary`` only weakens the *key*: a controller holding
    a strong reference to its own backend would keep that key reachable
    through the dict's value, so closing a viewer would leak its
    backend, its plotter and the whole scene behind it.
    """
    import gc

    from apeGmsh.viewers.core._legend import controller_for

    sink = _Sink()
    controller_for(sink)
    ref = weakref.ref(sink)
    del sink
    gc.collect()
    assert ref() is None, "the controller pinned its own backend alive"


def test_controller_survives_a_collected_backend():
    """A mutator after the viewport is gone must not raise."""
    import gc

    sink = _Sink()
    ctl = LegendController(sink)
    ctl.register(("", "a"), "L0", _Handle("L0"), lut=LutSpec())
    del sink
    gc.collect()
    ctl.set_fmt(("", "a"), "%.1f")      # must be a silent no-op
    ctl.set_visible(("", "a"), False)


def test_shared_legend_repoints_after_its_first_source_detaches():
    """The bar must never stay bound to a removed layer's mapper.

    Two diagrams share a legend; the one whose handle the bar was built
    from detaches. The legend survives — and has to rebind to a handle
    that still exists.
    """
    ctl, sink = _controller()
    key = ("", "stress_vm")
    ha, hb = _Handle("layerA"), _Handle("layerB")
    ctl.register(key, "layerA", ha, lut=LutSpec())
    ctl.register(key, "layerB", hb, lut=LutSpec())
    assert ctl.entry(key).handle is ha

    ctl.unregister("layerA")
    assert ctl.entry(key).handle is hb
    assert "layerA" not in ctl.entry(key).sources


def test_relabelled_geometry_does_not_orphan_the_old_legend():
    """A layer re-registering under a new key leaves its old one.

    The concurrent-geometry resolver can answer differently between two
    attaches (a rename, or a second geometry becoming visible). Without
    this, the first legend keeps a source that will never detach and so
    never retires — a bar stuck on screen forever.
    """
    ctl, sink = _controller()
    ctl.register(("Geo 1", "vm"), "L0", _Handle("L0"), lut=LutSpec())
    ctl.register(("Geo 2", "vm"), "L0", _Handle("L0"), lut=LutSpec())

    assert [e.key for e in ctl.entries()] == [("Geo 2", "vm")]
    assert len(sink.bars) == 1


@pytest.mark.parametrize("viewport", [(300, 100), (300, 90), (240, 60), (200, 40)])
def test_degenerate_viewport_never_yields_a_negative_bar(viewport):
    """The reserved title band may not eat the bar it sits above.

    In a viewport too short for the content the box gets clamped below
    what the content asked for; subtracting a full title band from it
    produced a negative height and a title anchored off-screen.
    """
    ctl, _ = _controller(viewport)
    ctl.default_font_scale = MAX_FONT_SCALE
    _register(ctl, "displacement_magnitude", vertical=False,
              vmin=-1234.5, vmax=1234.5)
    spec = ctl.specs()[0]
    assert spec.extent[0] > 0 and spec.extent[1] > 0, spec.extent
    assert 0.0 <= spec.anchor[0] <= 1.0 and 0.0 <= spec.anchor[1] <= 1.0
    assert spec.title_anchor is not None
    assert 0.0 <= spec.title_anchor[0] <= 1.0
    assert 0.0 <= spec.title_anchor[1] <= 1.0


# =====================================================================
# L-STICKY — a hand-placed legend survives everything else (ADR 0081 L1)
# =====================================================================

def test_hand_placed_anchor_survives_every_other_change():
    """The regression test for the defect this ADR exists to kill.

    Pre-0081, every colormap change, clim autofit, step change and
    show/hide toggle destroyed the bar and re-created it from the style,
    silently discarding wherever the user had put it.
    """
    ctl, _ = _controller()
    entry = _register(ctl, "stress_vm")
    ctl.set_anchor(("", "stress_vm"), (0.62, 0.71))
    placed = entry.anchor
    assert placed == pytest.approx((0.62, 0.71))

    ctl.set_lut(("", "stress_vm"), LutSpec(name="turbo", vmin=-9.0, vmax=9.0))
    assert entry.anchor == pytest.approx(placed), "a recolour moved it"

    ctl.set_fmt(("", "stress_vm"), "%.5f")
    assert entry.anchor == pytest.approx(placed), "a format change moved it"

    ctl.set_visible(("", "stress_vm"), False)
    ctl.set_visible(("", "stress_vm"), True)
    assert entry.anchor == pytest.approx(placed), "a hide/show moved it"

    _register(ctl, "displacement_z")     # a second diagram arrives
    ctl.layout()
    assert entry.anchor == pytest.approx(placed), "a new legend moved it"

    ctl.unregister("layer::::displacement_z")
    assert entry.anchor == pytest.approx(placed), "a detach moved it"


def test_a_hand_placed_legend_is_skipped_by_slot_allocation():
    """An undocked legend must not consume a docked slot."""
    ctl, _ = _controller()
    _register(ctl, "a")
    _register(ctl, "b")
    ctl.set_anchor(("", "a"), (0.5, 0.5))
    ctl.layout()
    # 'b' is now the only docked legend, so it takes the first slot.
    assert ctl.entry(("", "b")).slot == 0
    assert ctl.entry(("", "a")).slot is None


# =====================================================================
# L-EVENTS — owners fire, exactly once (ADR 0056 Part 2)
# =====================================================================

class _Dispatcher:
    def __init__(self) -> None:
        self.fired: list = []

    def fire(self, kind, **kw) -> None:
        self.fired.append(kind)


def _with_dispatcher():
    ctl, sink = _controller()
    ctl.dispatcher = _Dispatcher()
    return ctl, sink, ctl.dispatcher


def test_every_mutator_fires_exactly_one_legend_changed():
    from apeGmsh.viewers.diagrams._dispatch import LEGEND_CHANGED

    ctl, _, disp = _with_dispatcher()
    _register(ctl, "a")
    assert disp.fired == [LEGEND_CHANGED]

    for call in (
        lambda: ctl.set_visible(("", "a"), False),
        lambda: ctl.set_visible(("", "a"), True),
        lambda: ctl.set_vertical(("", "a"), True),
        lambda: ctl.set_fmt(("", "a"), "%.1f"),
        lambda: ctl.set_font_scale(("", "a"), 1.4),
        lambda: ctl.set_lut(("", "a"), LutSpec(name="turbo")),
        lambda: ctl.set_anchor(("", "a"), (0.3, 0.3)),
        lambda: ctl.redock(("", "a")),
        lambda: ctl.unregister("layer::::a"),
    ):
        disp.fired.clear()
        call()
        assert disp.fired == [LEGEND_CHANGED], call


def test_a_no_op_write_fires_nothing():
    """Idempotent-skip per call (ADR 0056 Part 2)."""
    ctl, _, disp = _with_dispatcher()
    _register(ctl, "a", fmt="%.3g")
    disp.fired.clear()
    ctl.set_fmt(("", "a"), "%.3g")
    ctl.set_visible(("", "a"), True)
    ctl.set_vertical(("", "a"), False)
    assert disp.fired == []


def test_default_font_scale_fires_and_moves_every_following_legend():
    from apeGmsh.viewers.diagrams._dispatch import LEGEND_CHANGED

    ctl, _, disp = _with_dispatcher()
    entry = _register(ctl, "a")
    before = entry.extent
    disp.fired.clear()

    ctl.default_font_scale = 2.0
    assert disp.fired == [LEGEND_CHANGED]
    assert entry.extent[0] > before[0] and entry.extent[1] > before[1]

    # A legend with its own override ignores the viewer default.
    ctl.set_font_scale(("", "a"), 1.0)
    fixed = ctl.entry(("", "a")).extent
    ctl.default_font_scale = 3.0
    assert ctl.entry(("", "a")).extent == pytest.approx(fixed)


def test_controller_without_a_dispatcher_still_reconciles():
    """Unit-test controllers have no director; mutating must not raise."""
    ctl, sink = _controller()
    _register(ctl, "a")
    ctl.set_fmt(("", "a"), "%.1f")
    assert sink.bars["a"].fmt == "%.1f"


# =====================================================================
# Viewport resize (ADR 0081 L1, adversarial finding A6)
# =====================================================================

def test_relayout_after_resize_holds_the_box_pixel_size():
    """Boxes are normalized but derived from pixel font metrics.

    Left alone across a resize, the box scales with the window while the
    text stays at a fixed point size — so a shrinking window eventually
    breaks the fit guarantee.
    """
    ctl, sink = _controller((1600, 1000))
    _register(ctl, "displacement_magnitude", vmin=-1234.5, vmax=1234.5)
    e = ctl.entry(("", "displacement_magnitude"))
    px_before = (e.extent[0] * 1600, e.extent[1] * 1000)

    sink.viewport = (800, 500)
    ctl.refresh()
    px_after = (e.extent[0] * 800, e.extent[1] * 500)

    assert px_after[0] == pytest.approx(px_before[0], rel=0.02)
    assert px_after[1] == pytest.approx(px_before[1], rel=0.02)


def test_resize_hook_is_a_no_op_without_an_interactor():
    """Offscreen and headless backends have nothing to observe."""
    ctl, _ = _controller()
    assert ctl.install_resize_hook() is False


def test_resize_hook_observes_configure_event_once():
    """The hook is VTK's window-resize notification, installed once."""
    observed: list = []

    class _Iren:
        def AddObserver(self, event, cb):
            observed.append((event, cb))
            return len(observed)

    class _Backend(_Sink):
        def __init__(self):
            super().__init__()
            self.plotter = type(
                "P", (), {"iren": type("I", (), {"interactor": _Iren()})()},
            )()

    backend = _Backend()
    ctl = LegendController(backend)
    _register(ctl, "a")

    assert ctl.install_resize_hook() is True
    assert ctl.install_resize_hook() is True      # idempotent
    assert [e for e, _ in observed] == ["ConfigureEvent"]

    # Firing the observer re-lays-out against the new viewport.
    backend.viewport = (640, 400)
    observed[0][1]()
    e = ctl.entry(("", "a"))
    assert e.extent[0] * 640 == pytest.approx(
        _controller_box_px(ctl, e)[0], rel=0.02
    )


def _controller_box_px(ctl, entry):
    return box_px(entry, measure(entry, ctl.default_font_scale))


# =====================================================================
# Session round-trip (ADR 0081 L3)
# =====================================================================

def test_placement_round_trips_through_a_snapshot():
    """Save a placed legend, restore it into a fresh controller.

    Restore runs *before* the diagrams attach, so the snapshot has to be
    parked and claimed by the matching ``register``.
    """
    ctl, _ = _controller()
    _register(ctl, "stress_vm")
    ctl.set_anchor(("", "stress_vm"), (0.62, 0.71))
    ctl.set_font_scale(("", "stress_vm"), 1.75)
    ctl.set_fmt(("", "stress_vm"), "%.4f")
    ctl.set_vertical(("", "stress_vm"), False)
    saved = ctl.snapshot()
    # Not the literal 0.62 typed above: a wide horizontal legend at
    # 1.75x is clamped inward to stay in the viewport, and it is the
    # *contained* anchor that gets saved.
    assert saved[0]["anchor"] == ctl.entry(("", "stress_vm")).anchor

    fresh, _sink = _controller()
    fresh.restore(saved)                 # nothing registered yet
    assert fresh.entries() == []

    _register(fresh, "stress_vm")        # the diagram attaches
    e = fresh.entry(("", "stress_vm"))
    assert e.anchor == pytest.approx(saved[0]["anchor"])
    assert e.anchor[1] == pytest.approx(0.71)
    assert e.slot is None
    assert e.font_scale == pytest.approx(1.75)
    assert e.fmt == "%.4f"
    assert e.vertical is False


def test_a_docked_legend_restores_to_its_slot_not_its_anchor():
    """A session saved at one window size must lay out at another.

    A docked legend's anchor is a layout *output*; restoring it
    verbatim would strand the legend when the window changed size.
    """
    ctl, _ = _controller((1600, 1000))
    _register(ctl, "a")
    saved = ctl.snapshot()
    assert saved[0]["slot"] == 0

    fresh, _sink = _controller((640, 480))
    fresh.restore(saved)
    _register(fresh, "a")
    e = fresh.entry(("", "a"))
    assert e.slot == 0
    # Re-derived for the smaller viewport, not copied from the save.
    assert e.anchor != pytest.approx(tuple(saved[0]["anchor"]))
    assert e.anchor[0] + e.extent[0] <= 1.0


def test_restoring_a_legend_no_diagram_recreates_is_harmless():
    """A saved scale whose component is gone simply never wakes up."""
    ctl, sink = _controller()
    ctl.restore([{
        "geometry": "", "component": "gone", "vertical": True,
        "visible": True, "fmt": "%.3g", "font_scale": 2.0,
        "slot": None, "anchor": (0.3, 0.3),
    }])
    assert ctl.entries() == []
    assert sink.bars == {}


def test_restore_survives_malformed_snapshots():
    ctl, _ = _controller()
    ctl.restore([{"nope": 1}, None, {"geometry": "", "component": "a"}])
    _register(ctl, "a")
    assert ctl.entry(("", "a")) is not None


def test_unclaimed_placement_does_not_ambush_a_later_diagram():
    """Closing the restore window drops what the session did not recreate.

    Parked placement that outlives the restore would silently apply to a
    diagram the user adds by hand later, instead of giving it a fresh
    docked legend.
    """
    ctl, _ = _controller()
    ctl.restore([{
        "geometry": "", "component": "stress_vm",
        "slot": None, "anchor": (0.2, 0.8), "font_scale": 2.5,
    }])
    assert ctl.end_restore() == 1

    _register(ctl, "stress_vm")          # added long after the restore
    e = ctl.entry(("", "stress_vm"))
    assert e.slot == 0, "inherited a stale hand-placement"
    assert e.font_scale is None


def test_end_restore_keeps_placement_already_claimed():
    ctl, _ = _controller()
    ctl.restore([{
        "geometry": "", "component": "a", "slot": None, "anchor": (0.2, 0.8),
    }])
    _register(ctl, "a")                  # claims it inside the window
    assert ctl.end_restore() == 0
    assert ctl.entry(("", "a")).anchor == pytest.approx((0.2, 0.8))


# =====================================================================
# Web parity (ADR 0042 / ADR 0081) — the legend crosses the seam
# =====================================================================

def test_the_trame_backend_renders_legends_too():
    """Keeping the legend in the scene IR is what buys web parity.

    A Qt overlay would have been easier to build and would not exist on
    the web, in screenshots, or in export. ``TrameBackend`` inherits the
    projection unchanged — this asserts it rather than assuming it.
    """
    pytest.importorskip("pyvista")
    import pyvista as pv

    from apeGmsh.viewers.backends.trame import TrameBackend

    try:
        backend = TrameBackend(pv.Plotter(off_screen=True))
    except Exception:                              # pragma: no cover
        pytest.skip("no offscreen render context")

    mesh = pv.Sphere()
    mesh["v"] = mesh.points[:, 2]
    actor = backend.plotter.add_mesh(mesh, scalars="v", show_scalar_bar=False)

    ctl = LegendController(backend)
    ctl.register(
        ("", "displacement_z"), "L0", _PvHandle(actor),
        lut=LutSpec(vmin=-0.5, vmax=0.5),
    )
    spec = ctl.specs()[0]

    assert backend.viewport_size()[0] > 0
    bar = backend.plotter.scalar_bars["displacement_z"]
    assert bar.GetWidth() == pytest.approx(spec.extent[0], abs=1e-6)
    assert bar.GetPosition()[0] == pytest.approx(spec.anchor[0], abs=1e-6)
    # And the drag fast path works there too.
    assert backend.move_scalar_bar("displacement_z", spec) is True
    backend.plotter.close()


def test_horizontal_legend_removal_leaves_no_actors(pv_backend):
    """The separately drawn title actor is retired with its bar."""
    import pyvista as pv

    mesh = pv.Sphere()
    mesh["v"] = mesh.points[:, 2]
    actor = pv_backend.plotter.add_mesh(mesh, scalars="v", show_scalar_bar=False)

    before = pv_backend.plotter.renderer.GetActors2D().GetNumberOfItems()
    ctl = LegendController(pv_backend)
    ctl.register(
        ("", "vm"), "L0", _PvHandle(actor), lut=LutSpec(), vertical=False,
    )
    assert pv_backend.plotter.renderer.GetActors2D().GetNumberOfItems() > before

    ctl.unregister("L0")
    assert list(pv_backend.plotter.scalar_bars.keys()) == []
    assert (
        pv_backend.plotter.renderer.GetActors2D().GetNumberOfItems() == before
    )
