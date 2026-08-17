"""LegendController — the viewer's colour scales (ADR 0081).

A colour scale is a **view annotation owned by the viewer**, not a
property of the diagram that happens to be coloured. This module owns
that state and the only layout algorithm allowed to place it.

Before ADR 0081 the scale lived in five places at once (the diagram's
style dataclass, the mixin's ``_runtime_*`` overrides, a dead
``LUT.show_scalar_bar`` flag, the VTK actor's geometry, and the
interactive widget's own copy), was keyed two different ways, and had
two divergent layout implementations. The consequences were visible:
N diagrams stacked N bars in exactly one spot, the "size" knob shrank
the box but not the 18 pt text inside it, and vertical bars pushed
their title off the right edge of the window.

Three rules replace all of that:

* **One entry per coloured array.** ``LegendEntry`` is keyed by
  :class:`LegendKey` — ``(geometry, component)`` — so two diagrams
  showing the same component share one scale instead of drawing two
  on top of each other. The geometry half keeps concurrent geometries
  (ADR 0058) apart without the title-prefix hack; the *title* collapses
  to the bare component while only one geometry is on screen.
* **The box is derived from its content.** :func:`measure` sizes the
  box from the font metrics, the tick count and the widest formatted
  label — never the other way round. "Too small for its own text" and
  "title off-viewport" stop being clamped after the fact and become
  unrepresentable, and ``font_scale`` becomes the honest knob the old
  ``size`` multiplier only pretended to be.
* **The controller places every legend.** Docked entries get a slot
  along their edge; a user-placed entry (``slot is None``, set by the
  ADR 0081 L2 interactor) keeps its anchor and is only clamped for
  containment. pyvista's own slot allocator cannot help here — it is
  permanently disabled the moment an explicit position is passed,
  which the backend always does.

The controller is a plain object (no Qt), so it is fully testable
headless. One controller exists per render target, keyed weakly by
backend in :func:`controller_for` — the backend is per-viewer by
construction, which gives every viewport exactly one owner without
threading an injection parameter through the diagram lifecycle.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Optional
from weakref import WeakKeyDictionary


#: Spec fields whose change can be applied to a live bar actor. Anything
#: outside this set (a new LUT, a new title, a flipped orientation)
#: needs the bar rebuilt.
_GEOMETRY_FIELDS = ("anchor", "extent", "title_pt", "label_pt", "title_anchor")


def _geometry_only(before: Any, after: Any) -> bool:
    """Whether two specs differ only in fields a live actor can absorb."""
    for name in ("key", "title", "lut", "fmt", "vertical", "n_labels"):
        if getattr(before, name) != getattr(after, name):
            return False
    return True


def _weak(obj: Any) -> Any:
    """``weakref.ref(obj)``, or a plain thunk when ``obj`` forbids it."""
    if obj is None:
        return None
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


# =====================================================================
# Layout constants
# =====================================================================

#: Base point size at ``font_scale == 1.0``. Deliberately below
#: pyvista's stock 18 pt: at 18 pt a legend's text dominates its own
#: box, which is half of why the old scale read as "enormous".
BASE_TITLE_PT = 14

#: Tick labels sit a step below the title.
LABEL_PT_RATIO = 0.85

#: Horizontal advance per character as a fraction of the point size.
#: A deliberate over-estimate for VTK's default sans face — the box
#: must never be *narrower* than its text, and a slightly generous box
#: is invisible where a clipped label is not. Pinned against measured
#: VTK text extents by ``test_legend_layout.py::test_advance_estimate_
#: brackets_measured_text``.
CHAR_ADVANCE = 0.62

#: Line height as a fraction of the point size.
LINE_HEIGHT = 1.35

#: Minimum thickness of the colour ramp itself, in pixels. Below this
#: the ramp reads as a hairline (the pre-0081 rendering).
MIN_RAMP_PX = 14

#: Padding inside the box and between stacked legends, in pixels.
PAD_PX = 6
GUTTER_PX = 14
MARGIN_PX = 16

#: Clamp for the user-facing size knob.
MIN_FONT_SCALE = 0.5
MAX_FONT_SCALE = 3.0


# =====================================================================
# Value types
# =====================================================================

#: ``(geometry, component)``. ``geometry`` is ``""`` when the viewer
#: has no concurrent-geometry discriminator for the diagram.
LegendKey = tuple[str, str]


def key_str(key: LegendKey) -> str:
    """Stable string form of a key — the backend's dict key."""
    geometry, component = key
    return f"{geometry}\x1f{component}" if geometry else component


@dataclass
class LegendEntry:
    """One colour scale. Mutable — the controller owns it."""

    key: LegendKey
    component: str
    lut: Any                      # scene_ir.LutSpec
    fmt: str = "%.3g"
    vertical: bool = True
    font_scale: Optional[float] = None   # None = controller default
    visible: bool = True
    n_labels: int = 5
    #: Docked slot index, or ``None`` once the user has placed it by
    #: hand (ADR 0081 L2). A ``None`` slot means "honour ``anchor``".
    slot: Optional[int] = 0
    #: Lower-left corner in normalized viewport coords. Written by the
    #: layout for docked entries; authoritative for undocked ones.
    anchor: tuple[float, float] = (0.0, 0.0)
    #: Resolved (width, height) in normalized viewport coords.
    extent: tuple[float, float] = (0.0, 0.0)
    #: ``layer_id -> LayerHandle`` for every layer feeding this legend.
    #: The entry dies with the last one.
    #:
    #: A mapping rather than a set plus one "primary" handle: when two
    #: diagrams share a legend and the first detaches, a remembered
    #: primary would leave the bar bound to the mapper of a layer that
    #: no longer exists.
    sources: dict = field(default_factory=dict)

    @property
    def handle(self) -> Any:
        """A live source handle to bind the bar's mapper to."""
        for handle in self.sources.values():
            return handle
        return None


@dataclass(frozen=True)
class TextMetrics:
    """Estimated pixel extents for one legend's text at a given size."""

    title_pt: int
    label_pt: int
    title_w: float
    label_w: float
    line_h: float


# =====================================================================
# Content-derived measurement
# =====================================================================

def text_width_px(text: str, pt: float) -> float:
    """Estimated rendered width of ``text`` at ``pt`` points."""
    return CHAR_ADVANCE * float(pt) * len(text)


def tick_labels(entry: LegendEntry) -> list[str]:
    """The tick labels VTK will draw for ``entry``, as formatted text.

    Mirrors ``vtkScalarBarActor``: ``n_labels`` values evenly spaced
    across the LUT range, rendered through the entry's printf format.
    A bad format string degrades to ``repr`` rather than raising —
    the user is typing into that field live.
    """
    vmin = float(getattr(entry.lut, "vmin", 0.0))
    vmax = float(getattr(entry.lut, "vmax", 1.0))
    n = max(int(entry.n_labels), 2)
    step = (vmax - vmin) / (n - 1)
    out = []
    for i in range(n):
        value = vmin + i * step
        try:
            out.append(entry.fmt % value)
        except (TypeError, ValueError):
            out.append(f"{value:.3g}")
    return out


def measure(entry: LegendEntry, font_scale: float) -> TextMetrics:
    """Font sizes + text extents for ``entry`` at ``font_scale``."""
    scale = min(max(float(font_scale), MIN_FONT_SCALE), MAX_FONT_SCALE)
    title_pt = max(6, int(round(BASE_TITLE_PT * scale)))
    label_pt = max(5, int(round(BASE_TITLE_PT * LABEL_PT_RATIO * scale)))
    labels = tick_labels(entry)
    return TextMetrics(
        title_pt=title_pt,
        label_pt=label_pt,
        title_w=text_width_px(entry.component, title_pt),
        label_w=max(text_width_px(t, label_pt) for t in labels),
        line_h=LINE_HEIGHT * label_pt,
    )


def title_band_px(entry: LegendEntry, m: TextMetrics) -> float:
    """Height reserved *above* a horizontal bar for the title.

    Zero for vertical legends, where ``vtkScalarBarActor``'s own title
    placement is correct. Horizontal bars get a band because VTK's is
    not: it centres the title inside the tick-label band, so the title
    sits on top of the middle label — at every box size, with or
    without ``UnconstrainedFontSize``. The controller reserves the band
    and the backend draws the title into it.
    """
    return 0.0 if entry.vertical else LINE_HEIGHT * m.title_pt + PAD_PX


def box_px(entry: LegendEntry, m: TextMetrics) -> tuple[float, float]:
    """The box ``(w, h)`` in pixels that fits ``entry``'s content.

    ``vtkScalarBarActor`` lays its tick labels and colour ramp out
    *inside* whatever box it is given; if the box is too small it does
    not shrink the text, it overlaps it and starves the ramp to a
    hairline. So the box is computed from the content it has to hold:

    * vertical — the ramp runs up the left with the labels to its right
      and VTK's own title centred above, so the width is
      ``ramp + gap + widest label`` (and at least the title), and the
      height must give every label its own line.
    * horizontal — the tick labels sit above the ramp, sharing the box;
      the title lives in the separate band :func:`title_band_px`
      reserves above it. The width must fit every label side by side.

    Horizontal tick labels are centred on their tick, so the outermost
    two hang half a label past each end of the ramp — the width budget
    carries that overhang, which is what used to push "0.5" off the
    edge of the window.
    """
    ramp = max(MIN_RAMP_PX, 0.9 * m.label_pt)
    if entry.vertical:
        w = max(m.title_w, ramp + PAD_PX + m.label_w) + 2 * PAD_PX
        h = (
            LINE_HEIGHT * m.title_pt + PAD_PX
            + max(entry.n_labels, 2) * m.line_h * 1.6
        )
    else:
        labels_w = max(entry.n_labels, 2) * (m.label_w + PAD_PX)
        w = max(m.title_w, labels_w + m.label_w) + 2 * PAD_PX
        h = title_band_px(entry, m) + ramp + PAD_PX + m.line_h
    return w, h


# =====================================================================
# Controller
# =====================================================================

class LegendController:
    """Owns every colour scale in one viewport (ADR 0081 Part 1).

    Diagrams call :meth:`register` at attach and :meth:`unregister` at
    detach and are otherwise not involved: they do not hold legend
    state, do not compute geometry, and do not talk to the backend
    about bars. Every mutator reconciles — it recomputes the layout and
    pushes only what changed, so a legend the user has placed is never
    moved by an unrelated recolour.
    """

    def __init__(self, backend: Any) -> None:
        # Weak: :func:`controller_for` keys a WeakKeyDictionary by
        # backend, and a strong reference here would make the value keep
        # its own key alive — the backend, its plotter and the whole
        # scene would outlive the viewer that owned them.
        self._backend_ref = _weak(backend)
        self._entries: dict[LegendKey, LegendEntry] = {}
        self._order: list[LegendKey] = []
        self._applied: dict[str, Any] = {}
        #: Session snapshots parked until their legend registers
        #: (restore runs before the diagrams attach).
        self._pending: dict[LegendKey, dict] = {}
        self._default_font_scale: float = 1.0
        #: Injected by ``DiagramRegistry.bind`` (ADR 0056 Part 2 — owner
        #: mutators fire their own dispatcher events). ``None`` for a
        #: controller built outside a director, i.e. in unit tests.
        self.dispatcher: Any = None
        self._resize_tag: Any = None

    @property
    def default_font_scale(self) -> float:
        """Text size every legend that has no override of its own uses."""
        return self._default_font_scale

    @default_font_scale.setter
    def default_font_scale(self, scale: float) -> None:
        clamped = min(max(float(scale), MIN_FONT_SCALE), MAX_FONT_SCALE)
        if clamped == self._default_font_scale:
            return
        self._default_font_scale = clamped
        self._reconcile_and_fire()

    @property
    def _backend(self) -> Any:
        """The render target, or ``None`` once it has been collected."""
        return self._backend_ref() if self._backend_ref is not None else None

    # -- registration --------------------------------------------------

    def register(
        self,
        key: LegendKey,
        layer_id: str,
        handle: Any,
        *,
        lut: Any,
        fmt: str = "%.3g",
        vertical: bool = True,
        font_scale: Optional[float] = None,
        visible: bool = True,
    ) -> LegendEntry:
        """Bind a layer to the legend for ``key``, creating it on miss.

        The style-derived arguments are **initial preferences**: they
        seed a new entry and are ignored when one already exists, so a
        second diagram joining an existing scale cannot silently
        redecorate it (nor can a re-attach undo the user's edits).
        """
        # A layer that reappears under a different key — its geometry was
        # renamed, or the concurrent-geometry resolver changed its answer
        # between attaches — must leave its old legend, or that legend
        # keeps a source that will never detach and never retires.
        self._drop_source(layer_id, except_key=key)
        entry = self._entries.get(key)
        if entry is None:
            entry = LegendEntry(
                key=key,
                component=key[1],
                lut=lut,
                fmt=fmt,
                vertical=bool(vertical),
                font_scale=font_scale,
                visible=bool(visible),
                slot=len(self._order),
            )
            self._entries[key] = entry
            self._order.append(key)
        entry.sources[layer_id] = handle
        # A session may have restored placement for this legend before
        # any diagram existed to create it (ADR 0081 L3).
        if key in self._pending:
            self._apply_pending()
        self._reconcile_and_fire()
        return entry

    def unregister(self, layer_id: str) -> None:
        """Drop ``layer_id`` as a source; retire empty legends."""
        self._drop_source(layer_id)
        self._reconcile_and_fire()

    def _drop_source(
        self, layer_id: str, *, except_key: "Optional[LegendKey]" = None,
    ) -> None:
        """Remove ``layer_id`` from every legend but ``except_key``."""
        for key in list(self._order):
            entry = self._entries.get(key)
            if entry is None or key == except_key:
                continue
            if entry.sources.pop(layer_id, None) is None:
                continue
            if not entry.sources:
                self._entries.pop(key, None)
                self._order.remove(key)
        self._renumber_slots()

    def entry(self, key: LegendKey) -> Optional[LegendEntry]:
        return self._entries.get(key)

    def entries(self) -> list[LegendEntry]:
        return [self._entries[k] for k in self._order]

    # -- mutators ------------------------------------------------------

    def set_visible(self, key: LegendKey, visible: bool) -> None:
        self._mutate(key, "visible", bool(visible))

    def set_vertical(self, key: LegendKey, vertical: bool) -> None:
        self._mutate(key, "vertical", bool(vertical))

    def set_fmt(self, key: LegendKey, fmt: str) -> None:
        self._mutate(key, "fmt", str(fmt))

    def set_font_scale(self, key: LegendKey, scale: float) -> None:
        clamped = min(max(float(scale), MIN_FONT_SCALE), MAX_FONT_SCALE)
        self._mutate(key, "font_scale", clamped)

    def set_lut(self, key: LegendKey, lut: Any) -> None:
        self._mutate(key, "lut", lut)

    def set_anchor(self, key: LegendKey, anchor: tuple[float, float]) -> None:
        """Place a legend by hand — undocks it (ADR 0081 L2)."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.anchor = (float(anchor[0]), float(anchor[1]))
        entry.slot = None
        self._renumber_slots()
        self._reconcile_and_fire()

    def redock(self, key: LegendKey) -> None:
        """Return a hand-placed legend to the automatic stack."""
        entry = self._entries.get(key)
        if entry is None or entry.slot is not None:
            return
        entry.slot = len(self._order)
        self._renumber_slots()
        self._reconcile_and_fire()

    def _mutate(self, key: LegendKey, field_name: str, value: Any) -> None:
        entry = self._entries.get(key)
        if entry is None or getattr(entry, field_name) == value:
            return
        setattr(entry, field_name, value)
        self._reconcile_and_fire()

    def _reconcile_and_fire(self) -> None:
        """Reconcile, then announce (ADR 0056 Part 2).

        The owner fires; call sites only call mutators. Reconciliation
        runs first so the dispatcher's coalesced render paints bars that
        are already correct.
        """
        self._reconcile()
        if self.dispatcher is None:
            return
        try:
            from ..diagrams._dispatch import LEGEND_CHANGED
            self.dispatcher.fire(LEGEND_CHANGED)
        except Exception:
            pass

    def _renumber_slots(self) -> None:
        """Compact docked slots to 0..n-1 in registration order."""
        n = 0
        for key in self._order:
            entry = self._entries[key]
            if entry.slot is not None:
                entry.slot = n
                n += 1

    # -- layout --------------------------------------------------------

    def viewport_px(self) -> tuple[int, int]:
        """The render target's size in pixels, with a sane fallback."""
        try:
            w, h = self._backend.viewport_size()
            if int(w) > 0 and int(h) > 0:
                return int(w), int(h)
        except Exception:
            pass
        return 1024, 768

    def layout(self) -> list[LegendEntry]:
        """Resolve ``anchor`` + ``extent`` for every visible entry.

        Horizontal legends dock along the bottom edge and stack upward;
        vertical ones dock along the right edge and stack leftward. A
        stack that would run past the opposite margin wraps into a
        second row/column rather than sliding off screen.
        """
        vw, vh = self.viewport_px()
        mx, my = MARGIN_PX / vw, MARGIN_PX / vh
        gx, gy = GUTTER_PX / vw, GUTTER_PX / vh

        # (stack offset along the dock normal, cross offset when wrapped)
        cursor = {False: [my, 0.0], True: [mx, 0.0]}

        visible = [e for e in self.entries() if e.visible]
        for entry in visible:
            m = measure(entry, self._scale_of(entry))
            w_px, h_px = box_px(entry, m)
            w, h = min(w_px / vw, 1.0 - 2 * mx), min(h_px / vh, 1.0 - 2 * my)
            entry.extent = (w, h)
            if entry.slot is None:
                entry.anchor = self._contain(entry.anchor, (w, h), mx, my)
                continue
            state = cursor[entry.vertical]
            if entry.vertical:
                if state[0] + w > 1.0 - mx:          # wrap to a new column
                    state[0], state[1] = mx, state[1] + 0.5
                x = 1.0 - state[0] - w
                y = max(my, (1.0 - h) * 0.5 - state[1])
                state[0] += w + gx
            else:
                if state[0] + h > 1.0 - my:          # wrap to a new row
                    state[0], state[1] = my, state[1] + 0.5
                x = min(max(mx, (1.0 - w) * 0.5 + state[1]), 1.0 - mx - w)
                y = state[0]
                state[0] += h + gy
            entry.anchor = self._contain((x, y), (w, h), mx, my)
        return visible

    @staticmethod
    def _contain(
        anchor: tuple[float, float],
        extent: tuple[float, float],
        mx: float,
        my: float,
    ) -> tuple[float, float]:
        """Clamp so the whole box stays inside the viewport margins."""
        w, h = extent
        x = min(max(float(anchor[0]), mx), max(mx, 1.0 - mx - w))
        y = min(max(float(anchor[1]), my), max(my, 1.0 - my - h))
        return x, y

    def _scale_of(self, entry: LegendEntry) -> float:
        return (
            self.default_font_scale
            if entry.font_scale is None else entry.font_scale
        )

    def _title_of(self, entry: LegendEntry) -> str:
        """Display title — the geometry prefix appears only when needed.

        Keying on ``(geometry, component)`` keeps concurrent geometries
        from colliding; showing the geometry in the title is only
        useful once more than one is on screen, so the prefix collapses
        below that threshold (ADR 0081 open question 1).
        """
        geometries = {k[0] for k in self._order if self._entries[k].visible}
        geometries.discard("")
        if len(geometries) > 1 and entry.key[0]:
            return f"{entry.key[0]} — {entry.component}"
        return entry.component

    # -- reconciliation ------------------------------------------------

    def specs(self) -> "list[Any]":
        """The ``ScalarBarSpec``s the backend should currently show."""
        from ..scene_ir import ScalarBarSpec

        _vw, vh = self.viewport_px()
        out = []
        for entry in self.layout():
            m = measure(entry, self._scale_of(entry))
            x, y = entry.anchor
            w, h = entry.extent
            # Horizontal legends carry their title in a reserved band
            # above the bar; the bar itself gets what is left. The band
            # is capped so it can never eat the bar: in a viewport too
            # short for the content, `layout` clamps the box below what
            # the content asked for, and an uncapped band would then
            # hand the backend a negative height and a title anchored
            # off-screen.
            band = min(title_band_px(entry, m) / vh, max(0.0, h - MIN_RAMP_PX / vh))
            # The upper clamp is itself floored at zero: a viewport
            # shorter than one line of title would otherwise produce a
            # negative bound and push the anchor off the bottom.
            title_top = max(0.0, 1.0 - LINE_HEIGHT * m.title_pt / vh)
            title_anchor = (
                None if entry.vertical
                else (
                    min(max(x + w * 0.5, 0.0), 1.0),
                    min(max(y + h - band + PAD_PX / vh, 0.0), title_top),
                )
            )
            out.append(ScalarBarSpec(
                key=key_str(entry.key),
                title=self._title_of(entry),
                lut=entry.lut,
                fmt=entry.fmt,
                vertical=entry.vertical,
                anchor=(x, y),
                extent=(w, h - band),
                title_pt=m.title_pt,
                label_pt=m.label_pt,
                n_labels=entry.n_labels,
                title_anchor=title_anchor,
            ))
        return out

    def _reconcile(self) -> None:
        """Push the current layout to the backend, diff-shaped."""
        if self._backend is None:
            return
        wanted = {s.key: s for s in self.specs()}
        for stale in [k for k in self._applied if k not in wanted]:
            try:
                self._backend.remove_scalar_bar(stale)
            except Exception:
                pass
            self._applied.pop(stale, None)
        for bar_key, spec in wanted.items():
            applied = self._applied.get(bar_key)
            if applied == spec:
                continue
            entry = self._entries[self._key_of(bar_key)]
            if entry.handle is None:
                continue
            if applied is not None and _geometry_only(applied, spec):
                # A drag is a geometry change per mouse-move event;
                # rebuilding the actor each time flickers and lags.
                mover = getattr(self._backend, "move_scalar_bar", None)
                if mover is not None and mover(bar_key, spec):
                    self._applied[bar_key] = spec
                    continue
            try:
                self._backend.add_scalar_bar(entry.handle, spec)
            except Exception:
                continue
            self._applied[bar_key] = spec

    def _key_of(self, bar_key: str) -> LegendKey:
        for key in self._order:
            if key_str(key) == bar_key:
                return key
        raise KeyError(bar_key)

    # -- session persistence (ADR 0081 L3) -----------------------------

    def snapshot(self) -> "list[dict]":
        """Placement of every legend, for the viewer session.

        Plain dicts keyed by ``(geometry, component)`` — the entry key,
        which is what survives a re-attach; layer ids do not.
        """
        return [
            {
                "geometry": e.key[0],
                "component": e.key[1],
                "vertical": bool(e.vertical),
                "visible": bool(e.visible),
                "fmt": e.fmt,
                "font_scale": e.font_scale,
                "slot": e.slot,
                "anchor": tuple(e.anchor),
            }
            for e in self.entries()
        ]

    def restore(self, snapshots: "Any") -> None:
        """Re-apply saved placement. Unknown legends are remembered.

        Restore runs before the diagrams attach, so the entries do not
        exist yet: the snapshot is parked and consumed by the matching
        :meth:`register`. A legend the session names but the restored
        diagrams never create simply never wakes up.

        A **docked** legend restores to its slot and lets the layout
        place it, so a session saved on one window size still lays out
        correctly on another; only a hand-placed one restores its anchor.

        Call :meth:`end_restore` once the session's diagrams have
        attached; anything still parked belongs to a legend the session
        named and the restore did not recreate.
        """
        self._pending = {}
        for raw in snapshots or ():
            try:
                key = (str(raw["geometry"]), str(raw["component"]))
            except Exception:
                continue
            self._pending[key] = raw
        self._apply_pending()
        self._reconcile_and_fire()

    def end_restore(self) -> int:
        """Close the restore window; drop unclaimed placement.

        Without this the parked snapshots live forever, and a diagram
        the user adds by hand an hour later would silently inherit a
        placement from a session restore instead of getting a fresh
        docked legend. Returns how many were dropped.
        """
        dropped = len(self._pending)
        self._pending = {}
        return dropped

    def _apply_pending(self) -> None:
        """Push any parked snapshot onto a legend that now exists."""
        for key, raw in list(getattr(self, "_pending", {}).items()):
            entry = self._entries.get(key)
            if entry is None:
                continue
            entry.vertical = bool(raw.get("vertical", entry.vertical))
            entry.visible = bool(raw.get("visible", entry.visible))
            entry.fmt = str(raw.get("fmt", entry.fmt))
            scale = raw.get("font_scale")
            entry.font_scale = None if scale is None else float(scale)
            slot = raw.get("slot")
            entry.slot = None if slot is None else int(slot)
            if entry.slot is None:
                anchor = raw.get("anchor") or entry.anchor
                entry.anchor = (float(anchor[0]), float(anchor[1]))
            self._pending.pop(key, None)
        self._renumber_slots()

    def refresh(self) -> None:
        """Re-run the layout against the current viewport.

        Legend boxes are stored in normalized viewport coordinates but
        are *derived* from pixel font metrics, so a resize invalidates
        them: left alone, the box scales with the window while its text
        stays at a fixed point size, and shrinking the window far enough
        breaks the fit guarantee. :func:`install_resize_hook` calls this.
        """
        self._reconcile_and_fire()

    def install_resize_hook(self) -> bool:
        """Re-layout whenever the render window is resized.

        Returns whether a hook was installed — ``False`` for headless
        and offscreen backends, which have no interactor to observe.
        Idempotent.
        """
        if self._resize_tag is not None:
            return True
        backend = self._backend
        try:
            iren = backend.plotter.iren.interactor
        except Exception:
            return False
        if iren is None:
            return False
        try:
            # ConfigureEvent is VTK's window-resize notification. The
            # observer never aborts — resizing is nobody's exclusive
            # gesture.
            self._resize_tag = iren.AddObserver(
                "ConfigureEvent", lambda *_: self.refresh(),
            )
        except Exception:
            return False
        return True


# =====================================================================
# One controller per render target
# =====================================================================

_CONTROLLERS: "WeakKeyDictionary[Any, LegendController]" = WeakKeyDictionary()


def controller_for(backend: Any) -> LegendController:
    """The :class:`LegendController` owning ``backend``'s viewport.

    Keyed weakly by backend because a backend *is* a viewport: one
    ``PyVistaQtBackend`` wraps one plotter, and the registry wraps once
    at bind (ADR 0042 R-B.final). That gives every viewport exactly one
    legend owner without threading an injection parameter through
    ``Diagram.attach`` — and gives headless diagram tests, which attach
    straight to an offscreen backend, the same owner the live viewer
    has.
    """
    controller = _CONTROLLERS.get(backend)
    if controller is None:
        controller = LegendController(backend)
        _CONTROLLERS[backend] = controller
    return controller


def adopt_controller(backend: Any, controller: Any) -> None:
    """Pre-bind ``controller`` as ``backend``'s legend owner.

    ADR 0098 S1: the session realize layer binds a *null* controller to
    its backend **before** any diagram attaches, so the per-diagram
    ``ScalarBarSupport`` registrations become no-ops and the session IR
    (``view.legends()``) stays the only author of colour scales
    (INV-LEGEND-1). Un-adopted backends are unaffected —
    :func:`controller_for` keeps building a real ``LegendController``
    on first miss exactly as before.
    """
    _CONTROLLERS[backend] = controller
