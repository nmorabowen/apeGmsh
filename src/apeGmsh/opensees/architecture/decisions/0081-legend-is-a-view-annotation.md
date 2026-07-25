# ADR 0081 — The scale is a view annotation, not a diagram property: `LegendController` + app-owned direct manipulation

**Status:** Accepted (2026-07-25) — design ratified by the user
2026-07-24; the full runway **L0–L3 has shipped** (PR #859).
Ratified forks: **(F1)** drag/resize is implemented in our own event
layer — `vtkScalarBarWidget` is dropped, not unblocked; **(F2)** the
runway is the full L0–L3 sequence, not a Phase-0 patch. The three open
questions carry their in-ADR proposals as ratified answers. This ADR
flips to Accepted at close-out with the per-slice PR numbers, per
house convention.

**Amended at L0 by implementation evidence** — see *L0 findings*
below: `vtkScalarBarActor` cannot lay out a horizontal title without
overlapping its own tick labels, so the controller reserves a title
band and the backend draws it; and legends now default to **vertical**.

Lands under ADR 0056 (viewer state & event contract) — the scale is
the most complete instance of the "one boolean, five homes" failure
that ADR 0056 Part 1 exists to outlaw, and it predates the contract.
Constrained by ADR 0042 (the legend must cross the render seam, so it
cannot be a Qt overlay).

## Context

The colour scale in the results viewer has been reported as broken
since it shipped: too large, not resizable to anything usable, not
movable despite presenting a drag handle. Every previous attempt
(#61, #783, ADR 0042 R-B Wave 1/2, ADR 0058 S2b) touched the same
per-diagram mixin and left the user-visible behaviour unchanged. A
July 2026 audit of the whole chain — mixin, backend, pyvista, VTK,
the pick engine, and rendered pixels — found five *independent*
defects and one root cause that regenerates them.

### The measured defects

**D1 — The drag handle is dead by construction.** Every bar is
created with `interactive=True`
([`pyvista_qt.py:400`](../../../viewers/backends/pyvista_qt.py),
[`_scalar_bar_support.py:266`](../../../viewers/diagrams/_scalar_bar_support.py)),
which builds a `vtkScalarBarWidget`. That widget observes the
interactor at VTK priority **0.5** (measured on VTK 9.5.2). The pick
backend aborts **every** plain left-button press at priority **10**:

```python
# _pyvista_pick.py:185
self._tags["lmb_press"] = iren.AddObserver("LeftButtonPressEvent", self._on_lmb_press, 10.0)
# _pyvista_pick.py:273 — unconditional
self._abort(caller, self._tags["lmb_press"])
```

The widget therefore never receives a press. The handle is drawn and
is physically unreachable — the exact "there seems to be a handle"
report. All three viewers install the pick engine, so all three are
affected.

**D2 — Even a working drag would be erased.** Any colormap or clim
mutation destroys and re-creates the bar
([`_scalar_color_support.py:186`](../../../viewers/diagrams/_scalar_color_support.py)),
and re-creation recomputes geometry from the spec
([`pyvista_qt.py:396-404`](../../../viewers/backends/pyvista_qt.py)).
Nothing ever reads the widget's geometry back. Bar position is
write-only from the spec side; the widget's copy is a dead end.

**D3 — `size` is not a size control.** It multiplies the box only
([`pyvista_qt.py:466-468`](../../../viewers/backends/pyvista_qt.py))
while fonts stay pinned at the theme's 18 pt with
`UnconstrainedFontSize=True`. Measured at 1280×800: `size=0.35` gives
a 0.21 × 0.028 box whose 18 pt title collides with its tick labels;
`size=2.0` clamps to `width=0.95`, i.e. the whole window. No value in
the spinner's 0.2–4.0 range produces a small, legible scale.

**D4 — N legends stack in exactly one place.** The backend always
passes explicit `position_x`/`position_y`, which disables pyvista's
slot allocator (`scalar_bars.py:412` — the allocator only runs when
either coordinate is `None`). Measured with three contour diagrams:
three bars at `(0.35, 0.05)`, titles rendered on top of each other.

**D5 — Vertical legends clip their own title** at the theme anchor
`position_x=0.9`; the centred title overflows the right viewport
edge.

### The root cause

The scale is modelled as a **property of each diagram**, delegated to
a widget the application cannot drive, with its geometry re-derived
from a style dataclass on every change. Under ADR 0056's taxonomy it
is intent state with **no owner** — it has five homes:

| Home | Written by | Read by |
|---|---|---|
| `style.show_scalar_bar` / `scalar_bar_vertical` / `scalar_bar_scale` / `fmt` | diagram construction, presets | the mixin's `_effective_*` |
| `_runtime_show_scalar_bar` / `_runtime_fmt` / `_runtime_bar_vertical` / `_runtime_bar_scale` | settings tab | the mixin's `_effective_*` |
| `LUT.show_scalar_bar` | [`_color_map_editor.py:339`](../../../viewers/ui/_color_map_editor.py) | **nobody** — verified dead |
| `vtkScalarBarActor` geometry | backend | nobody |
| widget representation geometry | VTK, on drag | nobody |

It also carries **two identity keys for one object**: `layer_id` on
the render-seam path and the title string on the legacy path — where
the title is computed on demand and can change under the diagram, a
hazard the code documents rather than fixes
([`_scalar_bar_support.py:196-199`](../../../viewers/diagrams/_scalar_bar_support.py):
"a bar created while only one geometry was visible keeps its
unprefixed title until re-created"). And **two divergent layout
implementations** for the same knobs — `_legacy_layout_kwargs`
(mixin; returns `{}` at defaults, hardcodes 0.6/0.08) versus
`_scalar_bar_layout` (backend; reads the theme, clamps) — so the
result depends on whether a diagram has been migrated to ADR 0042.

### Why the tests did not catch any of it

All 13 tests in `tests/viewers/test_contour_scalar_bar.py` assert
API-level facts — "a bar is registered under title X",
"`GetLabelFormat()` returns the fmt", "`GetWidth()` changed". None of
them fails for D1, D2, D3, D4 or D5. The suite verifies that we
called VTK, not that the user can read or move the scale.

## Decision

Six parts. Parts 1–3 relocate ownership and layout, Part 4 makes the
scale manipulable, Part 5 makes it durable, Part 6 makes all of it
provable.

### Part 1 — The legend is a view annotation owned by the viewer

A new `LegendController`, one per viewer, sibling of `LUTManager` in
`viewers/core/`. It owns `LegendEntry` records **keyed by the scalar
array being coloured**, not by diagram:

```python
@dataclass
class LegendEntry:
    key: LegendKey              # (geometry_id, component) — see Q1
    title: str                  # display only; may collapse (see Q1)
    lut_key: str                # → LUTManager
    fmt: str
    orientation: Literal["vertical", "horizontal"]
    anchor: tuple[float, float] # normalized viewport, APP-OWNED
    extent: tuple[float, float] # normalized viewport, derived (Part 3)
    font_scale: float           # single user-facing "size"
    visible: bool
    slot: int | None            # docked slot; None once user-dragged
    sources: set[str]           # layer_ids feeding this legend
```

Two diagrams colouring the same array share one legend — the sharing
`LUTManager` was built to support and the viewer wiring never used.

### Part 2 — Diagrams register; they do not own

`ScalarBarSupport` collapses to two calls: `legends.register(key,
handle, initial=self.spec.style)` at attach and
`legends.unregister(layer_id)` at detach. Deleted outright:

- all four `_runtime_bar_*` / `_runtime_show_scalar_bar` /
  `_runtime_fmt` fields,
- `LUT.show_scalar_bar` (dead state, no reader),
- `_bar_prefix_resolver` and the on-demand title computation — the
  entry key handles concurrent geometries structurally,
- `_legacy_layout_kwargs` **and** `_scalar_bar_layout`.

The style dataclasses keep their four fields, redocumented as
**initial preferences** consumed once at register time. Per ADR 0056
Part 1 the controller is the owner; style and widgets are
projections.

### Part 3 — One layout algorithm, and the box is derived from its content

The controller is the only code that computes legend geometry.

**Content-derived extent.** Today the box is chosen and the text is
left to fit (it doesn't — D3, D5). Invert it: the controller sizes
the box *from* the rendered content — `title_pt` and `label_pt`
(both scaled by the single `font_scale`), the number of ticks, and
the widest label under the active `fmt`. A horizontal legend's height
is `(title_pt + label_pt + padding) / viewport_px_h`; its width comes
from the label run. This makes "too small for its text" and "title
overflows the viewport" unrepresentable rather than clamped after the
fact, and turns `font_scale` into the honest knob the current `size`
pretends to be.

**Controller-side slots.** Because we always pass explicit positions,
pyvista's allocator is permanently off; the controller allocates
instead. Docked entries stack along their anchored edge with a fixed
gutter, wrapping to a second row/column at the viewport limit. D4
becomes structurally impossible.

**Viewport containment.** Anchors are resolved so
`anchor + extent + title_halo` stays inside `[0, 1]²` for both
orientations, which is D5's fix.

`ScalarBarSpec` widens to carry the resolved `anchor`, `extent`, and
font sizes; `RenderBackend.add_scalar_bar` becomes a dumb projection
of a controller-computed layout. Every backend — PyVista/Qt, trame,
a future ParaView export — gets correct legends from the same
numbers.

### Part 4 — Direct manipulation in our own event layer (fork F1)

`interactive=False` everywhere; no `vtkScalarBarWidget` is created.
A `LegendInteractor` installs on the interactor at priority **12**,
above navigation (11) and pick (10):

- On press it hit-tests the point against the visible entries' boxes
  in viewport coordinates. **Miss → return without aborting**, so
  camera and pick behaviour is bit-for-bit unchanged. **Hit → abort**,
  so navigation and pick never see a gesture aimed at the legend.
- Body drag moves the entry (snapping to viewport edges and corners
  within a tolerance; dropping on the home edge re-docks it to a
  slot). The corner opposite the anchor resizes, driving `font_scale`.
- Double-click re-docks. Hover sets the cursor.
- Right-click opens the context menu — Hide, Horizontal / Vertical,
  Format…, Rescale to data, Edit colour map… This is the
  **discoverable** surface; the settings-tab controls remain as the
  precise and scriptable path, rebuilt as projections of the
  controller.

Every gesture calls a controller mutator, and the mutator fires the
dispatcher event (ADR 0056 Part 2 — owners fire, call sites only
mutate). A new `LEGEND_CHANGED` kind enters the dispatcher matrix
with a render-only row.

### Part 5 — Anchors are durable

`LegendEntry` anchor / extent / orientation / visibility / font_scale
serialize into the existing viewer layout persistence, so a legend
placed once stays placed across sessions.

### Part 6 — Tests assert the picture

A new `tests/viewers/test_legend_layout.py`, and
`test_contour_scalar_bar.py` rewritten against the controller:

- **L-GEOM** — for a given window size, the rendered actor's box in
  **pixels** matches the controller's declared extent.
- **L-FIT** — at every `font_scale` in the allowed range, the
  formatted labels and title fit inside the box (font-metric
  assertion), and box + title stay inside the viewport for both
  orientations and every anchor.
- **L-NOOVERLAP** — N docked legends never intersect.
- **L-STICKY** — a move, then a colormap change, a step change, a
  clim autofit, and a show/hide toggle: the anchor is unchanged.
  This is D2's regression test.
- **L-EVENTS** — every controller mutator fires exactly one
  dispatcher event; `ui/` touches no bar actor (folds into ADR 0056's
  G-ARTIFACT guard).
- **L-PICK** — a press that misses every legend does not abort, and
  the existing navigation/pick suites stay green.

## Alternatives considered and rejected

1. **Unblock the VTK widget** — make the pick engine's abort
   conditional on a legend hit-test and keep `interactive=True`. A
   few lines, and the handle becomes live. Rejected as the primary
   fix: the geometry stays in a copy we neither own nor read back, so
   D2 survives untouched; it also couples the pick engine to VTK
   widget internals. Retained as the fallback if L2 proves expensive
   — it is compatible with Parts 1–3.
2. **Keep per-diagram bars, add slot allocation.** Fixes D4 alone and
   leaves five owners, two identity keys, and two layout
   implementations in place. This is the shape of every previous
   attempt.
3. **Render the legend as a Qt overlay widget** over the
   `QtInteractor`. Best-looking and fully controllable, but it breaks
   ADR 0042: the legend would not exist in the trame/web backend, in
   screenshots, or in export. The legend must live in the scene IR so
   every backend gets it — which is what Part 3 does.
4. **Document the limitation and move on.** Rejected: this is the
   single most-reported viewer defect, and the root cause keeps
   regenerating the symptoms.

## Consequences

- `ScalarBarSpec` widens (anchor, extent, font sizes) and `size`
  loses its current meaning. Backends become projections.
- The trame/web viewer inherits correct legends with no web-specific
  work.
- The mesh and model viewers adopt the controller — they install the
  same pick engine and carry the same defect.
- Net deletion in the diagram layer: the mixin loses its layout half,
  its runtime-override half, and the title-prefix hack.
- **Risk:** the priority-12 interceptor could swallow clicks meant
  for geometry beneath a legend. Mitigated by hit-testing the exact
  box and aborting only on a hit (L-PICK guards it).
- **Risk:** content-derived extents mean a `fmt` change resizes the
  box. Accepted — it is the correct behaviour, and re-docking keeps
  the layout tidy.

## L0 findings (implementation evidence, 2026-07-24)

Two things the design did not anticipate, both established by
rendering rather than by reading:

1. **`vtkScalarBarActor` cannot lay out a horizontal title.** It
   centres the title *inside* the tick-label band, so the title lands
   on top of the middle label. Measured invariant across box heights
   50–120 px, across `TextPosition` precede/succeed, and with
   `UnconstrainedFontSize` both on and off — it is not a
   parameterisation mistake, it is the actor's layout. Its *vertical*
   layout (title above, ramp left, labels right) is correct.
   **Decision:** the controller reserves a title band above horizontal
   bars (`title_band_px`) and publishes a `ScalarBarSpec.title_anchor`;
   the backend blanks the actor's title (there is no `DrawTitle` flag
   in the Python wrapping — `SetTitle("")` is the lever) and draws a
   `vtkTextActor` into the band. Vertical legends leave VTK's title
   alone. Covered by `test_horizontal_title_is_drawn_outside_the_bar`
   and `test_vertical_legend_lets_vtk_draw_its_own_title`.
2. **Legends default to vertical.** It is the orientation VTK gets
   right, it does not consume the bottom of the model view, and it
   matches the ParaView convention users arrive with. The style
   dataclasses' tri-state `scalar_bar_vertical` keeps `None` as "no
   preference", now resolving to vertical (`_default_vertical`).

Also folded in: the horizontal width budget carries the outermost tick
labels' half-label overhang past each end of the ramp — the specific
mechanism that pushed "0.5" off the edge of the window.

### L0 adversarial review

Six probes against the first cut; four were live defects, fixed with a
regression test each in `test_legend_layout.py`.

| # | Finding | Disposition |
|---|---|---|
| A1 | **Backend leak.** `controller_for` keys a `WeakKeyDictionary` by backend, but the controller held its backend *strongly* — so the dict's value kept its own key reachable and closing a viewer leaked the backend, the plotter and the whole scene. Measured: the backend survived dropping every strong reference. | Fixed — the controller holds a `weakref`; `test_controller_does_not_keep_its_backend_alive` (+ a mutate-after-collection no-op test). |
| A2 | **Stale mapper on a shared legend.** The entry remembered one "primary" handle. When two diagrams shared a legend and that one detached, the legend survived bound to the removed layer's mapper. | Fixed — `sources` became `layer_id -> handle` and `handle` a property over live sources; `test_shared_legend_repoints_after_its_first_source_detaches`. |
| A3 | **Negative bar height.** In a viewport too short for the content, `layout` clamps the box, and subtracting a full title band from the clamped height produced a *negative* extent and a title anchored off-screen. Measured from a 90 px-tall viewport down. | Fixed — the band is capped at `h - MIN_RAMP_PX` and the title anchor clamped into the viewport (with the clamp bound itself floored); `test_degenerate_viewport_never_yields_a_negative_bar`. |
| A5 | **Orphaned legend on re-key.** A layer re-registering under a different key (geometry renamed, or the concurrent-geometry resolver changing its answer between attaches) left its old entry holding a source that would never detach — a bar stuck on screen forever. | Fixed — `register` drops the layer from every other legend first; `test_relabelled_geometry_does_not_orphan_the_old_legend`. |
| A4 | Horizontal add/remove leaking the separately drawn title actor. | **Not a defect** — measured zero residual pyvista bars and zero 2D actors. Pinned anyway by `test_horizontal_legend_removal_leaves_no_actors`. |
| A9 | `set_show_scalar_bar` intent lost on a `VectorGlyph` with magnitude colours off (no legend key ⇒ the setter no-ops). | **Not a regression** — `use_magnitude_colors` has no live setter, so flipping it goes through rebuild-and-replace, which constructs a fresh diagram and discarded the old `_runtime_*` state pre-0081 too. |

Two further findings are genuinely L1's, not L0 patches, and are listed
in the runway below: the viewport-resize hook (A6) and the settings
tab misreporting size when an entry follows the controller default
(A10).

One quality defect in L0's own tests was caught and fixed: a
tautological assertion (`x == approx(x)`) in the hide-a-legend test —
precisely the vacuously-green shape this ADR exists to outlaw.

## Runway

- **L0 — layout core. SHIPPED.** `viewers/core/_legend.py`
  (`LegendController`, `LegendEntry`, content-derived `box_px`, slot
  allocation, viewport containment, `controller_for` keyed weakly by
  backend); `ScalarBarSpec` widened to a resolved layout and the
  PyVista backend reduced to a projection (`viewport_size()` added to
  the Protocol, `interactive=False`, `_scalar_bar_layout` and
  `_sync_bar_widget_geometry` deleted); all seven contour-bearing
  diagrams register/unregister instead of owning a bar;
  `_legacy_layout_kwargs` / `_ensure_scalar_bar` / `_scalar_bar_args`
  / `_scalar_bar_title` deleted.
  *Verified:* `tests/viewers/test_legend_layout.py` (42) — L-GEOM
  against a real offscreen render, L-FIT with a non-circular check of
  the width model against measured VTK text extents, L-NOOVERLAP at
  2/3/5 legends in both orientations; `test_contour_scalar_bar.py`
  rewritten against the controller (14); full viewer suite 1663
  passed. Screenshots confirm the three reported defects are gone.
- **L1 — ownership. SHIPPED.** `LUT.show_scalar_bar` and its setter
  deleted (the home nothing read); the colour-map editor's show-bar
  checkbox became a projection of the legend, reading and writing
  through the diagram. `LEGEND_CHANGED` joined the dispatcher matrix
  with an empty primitive row (the controller has already reconciled;
  the scene just needs painting) and every controller mutator fires it
  exactly once, with idempotent-skip per call (ADR 0056 Part 2).
  `DiagramRegistry.bind` wires the dispatcher and the resize hook,
  and exposes `registry.legends`. Both L0 review findings closed:
  **(A6)** `install_resize_hook` observes VTK's `ConfigureEvent` so a
  resize re-derives the boxes — without it a rendered legend kept its
  normalized box while its text stayed at a fixed point size, and a
  shrinking window eventually broke the fit guarantee; **(A10)**
  `_effective_bar_scale` now reports what actually renders, falling
  through to the controller default instead of the style. Open question
  3 fully answered: `Preferences.legend_font_scale` (a Labels-tab
  spinbox) drives `default_font_scale`, and a legend the user resized
  by hand keeps its own override.
  *Verified:* L-STICKY (a hand-placed anchor survives a recolour, a
  format change, a hide/show, a new legend arriving and a detach),
  L-EVENTS (nine mutators × exactly one fire; no-op writes fire
  nothing), and a resize test asserting the box holds its **pixel**
  size across 1600×1000 → 800×500. 60 tests in
  `test_legend_layout.py`; full suite 1689 passed.

  L1's own adversarial pass caught one defect in L1: the preferences
  subscription captured `self` on a module-level singleton and was
  never cancelled, pinning the whole viewer for the process lifetime
  and stacking one per viewer opened — the same class as A1. Now
  unsubscribed in the viewer's existing teardown loop.
- **L2 — direct manipulation. SHIPPED.**
  `viewers/core/_legend_interactor.py` — `LegendInteractor` observes
  press / move / release / right-click / leave at priority **12**,
  above navigation (11) and picking (10), and **aborts only on a
  hit**, so behaviour outside a legend is bit-for-bit unchanged.
  Body drag moves with edge snapping; a drop within `REDOCK_PX` of the
  home edge re-docks; the corner opposite the anchor resizes by
  driving `font_scale` (so the box follows the text and can never be
  dragged smaller than its own labels); double-click re-docks;
  hover sets the VTK cursor. Shift+LMB is never taken — that is
  navigation's orbit. Right-click hands the entry to an
  `on_context_menu` callback, keeping this module toolkit-free the way
  `install_navigation` does with `on_shift_click`; the viewer supplies
  the Qt menu (hide / orientation / reset size / snap back / number
  format… / rescale to data / edit colour map…).
  *Verified:* 26 gesture tests, incl. **L-PICK** (a miss aborts
  nothing for any of the four events) and a drag that survives a
  recolour — D2's failure mode in its original form. No regression in
  the pick and navigation suites (76 tests). Full suite 1712 passed.

  L2's adversarial pass found three defects in L2: a drag rebuilt the
  VTK bar actor **once per mouse-move event** (measured 40/40 — visible
  as flicker, felt as lag), fixed with a `move_scalar_bar` fast path on
  the backend for geometry-only changes, gated by a field comparison so
  a LUT / orientation / format change still rebuilds; a drag whose
  release landed outside the window never ended, leaving the legend
  glued to the cursor, fixed with a `LeaveEvent` observer; and hover
  re-asserted the same cursor on every mouse-move event.
- **L3 — durability and parity. SHIPPED.** Placement persists in the
  viewer session (`ViewerSession` schema **v7 → v8**, a `legends` block
  of `LegendSnapshot` records keyed by `(geometry, component)` — the
  entry key, because layer ids do not survive a re-attach). Restore runs
  *before* the diagrams attach, so snapshots are parked and each is
  claimed by the matching `register`; `end_restore` closes the window
  once the session's diagrams are in. A **docked** legend restores to
  its slot and is laid out fresh, so a session saved on one screen
  opens correctly on another; only a hand-placed one restores its
  anchor. Legacy sessions (no `legends` block) read as "everything
  docked at its default".

  **Web parity needed no code** — and that is the payoff for keeping
  the legend in the scene IR instead of building the Qt overlay
  alternative 3 rejected. `TrameBackend` inherits the projection
  unchanged; `test_the_trame_backend_renders_legends_too` asserts it
  rather than assuming it, including the L2 drag fast path.

  Docs: one prose section on the viewers page, per ADR 0079's
  single-home and course-not-archive rules. `mkdocs build --strict`
  clean.

  *Verified:* session round-trip (placed and docked), legacy-session
  load, malformed-snapshot tolerance, trame parity. Full suite 1721
  passed.

  L3's adversarial pass found one defect in L3: parked placement for a
  legend the restore never recreated lived forever, so a diagram the
  user added by hand an hour later would silently inherit a stale
  hand-placement instead of getting a fresh docked legend. Closed by
  `end_restore()`, called once `_apply_session` has attached everything.

## Open questions — all three resolved at ratification

1. **Legend key under concurrent geometries.** ADR 0058 S2b added a
   title-prefix resolver because two geometries can show the same
   component. Keying on `component` alone makes the key collide;
   keying on `(geometry_id, component)` makes it stable but yields
   two legends for what is often one physical quantity.
   **Resolved:** key is always `(geometry, component)`; the **title**
   collapses to the bare component while at most one geometry is
   visible. Stable key, clean display. Shipped in L0
   (`LegendController._title_of`).
2. **Does sharing a legend imply sharing a LUT?** Two diagrams on one
   component currently hold independent LUTs, so one shared legend
   would advertise a colour mapping only one of them uses.
   **Resolved:** yes — registering into an existing legend adopts that
   legend's LUT, and a diagram that wants its own mapping gets its own
   legend (explicit "unlink" in the L2 context menu). L0 ships the
   first half: `register` treats style-derived arguments as seeds and
   never redecorates a live entry. The LUT-mirror unification is L1.
3. **Is `font_scale` per legend or per viewer?** Per legend is more
   flexible; per viewer is what users actually want ("make the text
   bigger"). **Resolved:** a viewer-level default with a per-legend
   override, mirroring how label font sizes already work in
   `_PREF.current`. Shipped in L0 as
   `LegendController.default_font_scale` + `LegendEntry.font_scale`
   (`None` = follow the default); wiring the viewer preference into
   the default is L1.
