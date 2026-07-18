# Plotting Comparison — ParaView vs apeGmsh

Not a plan — an analysis. The data-pipeline and UI-panel sides were covered by the
earlier comparison rounds. This doc focuses specifically on the *rendering* layer: what
puts pixels on screen, who decides when to redraw, how transparency / antialiasing / LOD
are handled, and how scene objects (actors, scalar bars, axes, lights) are organized.

## The big picture

**ParaView's views actively orchestrate. apeGmsh's viewers passively host.**

In ParaView, every render goes through a three-pass cycle on the *view* — `Update`,
`UpdateLOD`, `Render` ([`vtkPVRenderView.h:1530-1606`](../../../ParaView/Remoting/Views/vtkPVRenderView.h)).
The view owns its representations, walks them, decides whether to use LOD geometry,
applies render passes (depth peeling, FXAA), and either does a `StillRender` (full
quality, antialiased, full geometry) or `InteractiveRender` (low quality during camera
drag). It is the *view*, not the data, that decides quality and timing.

In apeGmsh, the viewer is a `ViewerWindow` shell ([viewer_window.py:83](../../src/apeGmsh/viewers/ui/viewer_window.py)) hosting a PyVista
`QtInteractor` as a central widget. Every `Diagram` directly owns its actors and calls
`plotter.render()` ad hoc from its `update_to_step` ([_base.py:88-105](../../src/apeGmsh/viewers/diagrams/_base.py)). There is no view-level
"start of render frame / end of render frame" coordination. PyVista's plotter just
re-renders on demand at full quality every time.

That asymmetry has concrete consequences — listed below.

---

## Side-by-side breakdown

### 1. View / orchestration

| Aspect | ParaView | apeGmsh |
|---|---|---|
| View base class | `vtkPVView` (~400 LOC) + `vtkPVRenderView` (~1470 LOC) | `ViewerWindow` (~600 LOC) — passive host, not a base class |
| Owns reps? | Yes — `AddRepresentationInternal` | No — Diagrams own their own actors |
| Update lifecycle | `Update → UpdateLOD → Render` (three passes) | None — `plotter.render()` whenever |
| Per-frame work | View decides | Each Diagram decides |

**Cost to us:** When three Diagrams are on and the user drags time, each Diagram's
`update_to_step` calls `plotter.render()` independently. PyVista renders three times in a
row at full quality. There's no batching, no "wait until all diagrams have updated, then
render once."

---

### 2. Representation / Diagram

| Aspect | ParaView | apeGmsh |
|---|---|---|
| Unit | `vtkPVDataRepresentation` — owns mapper, `vtkPVLODActor`, LUT, visibility | `Diagram` — owns selector, style, actors list |
| Data ↔ render split? | Yes — source/filter produces data; representation renders it | No — Diagram is both (selector extracts, style + actors render) |
| Multiple reps per source? | Yes — one filter, many visualizations (wireframe + solid + edges) | No — one Diagram = one rendering |
| LOD actor? | Yes — `vtkPVLODActor` holds full-res + decimated geometry, swaps on demand | No — single actor, full geometry always |

**The architectural insight:** ParaView's representation is the clean abstraction
apeGmsh's `Diagram` is trying to be. We already have most of the protocol right —
`attach`/`update_to_step`/`detach` lifecycle, "mutate scalars in place; never re-add
actors" performance contract ([_base.py:96-104](../../src/apeGmsh/viewers/diagrams/_base.py)).
What's missing is the data ↔ render split. The selector lives on the Diagram
([_base.py:144](../../src/apeGmsh/viewers/diagrams/_base.py)); the actors live on the Diagram. One Diagram = one rendering = one
selector. That's why "show me deformed shape colored two different ways" requires two
Diagrams.

**This is the same insight that drives plans 04/06 indirectly** — it's just never been
named.

---

### 3. Render quality and lifecycle

| Aspect | ParaView | apeGmsh |
|---|---|---|
| Still vs interactive | `StillRender` (full quality) / `InteractiveRender` (LOD, no AA) auto-swapped on mouse drag | One render path; `enable_anti_aliasing` from preferences ([viewer_window.py:174-178](../../src/apeGmsh/viewers/ui/viewer_window.py)) |
| Depth peeling (transparent overlays) | `SetUseDepthPeeling(1) + SetMaximumNumberOfPeels(4)` ([`vtkPVRenderView.h:710`](../../../ParaView/Remoting/Views/vtkPVRenderView.h)) | Not configured |
| FXAA / SSAO / shadows | Available as switchable render passes | Whatever PyVista defaults to |
| LOD threshold | `SetLODRenderingThreshold(MB)` — drops to outline/decimation above threshold | None |

**Where this hurts us:** `results.viewer` with multiple transparent contour layers
suffers from z-fighting and incorrect occlusion ordering. The `Gauss marker` workaround
documented in memory (world-space spheres instead of pixel billboards) is *partly* a
depth-peeling workaround. Enabling depth peeling once on the renderer ([viewer_window.py:165-166](../../src/apeGmsh/viewers/ui/viewer_window.py))
costs ~4 hours and would solve this category of bug at the source.

Still vs interactive is a smaller win for our model sizes, but it would help when a
user rotates the camera with multiple heavy diagrams on. PyVista doesn't expose this
directly, but we can hook the interactor's mouse-press/release events and toggle a
"low quality" flag.

---

### 4. Annotations (scalar bars, axes, orientation)

| Aspect | ParaView | apeGmsh |
|---|---|---|
| Scalar bars | `vtkPVScalarBarActor` on the non-composited renderer; deduplicated by LUT | Per-Diagram, hand-placed |
| Orientation axes | `vtkPVAxesWidget` (always-corner orientation cube) | `plotter.add_axes(...)` ([viewer_window.py:179-188](../../src/apeGmsh/viewers/ui/viewer_window.py)) |
| Grid axes | `vtkPVGridAxes3DActor` with X/Y/Z labels | None |
| Time annotation | `vtkSMTextSourceRepresentationProxy` — text overlay tied to view time | None — time shown only in time-scrubber widget |
| Layering | Non-composited renderer (rendered after compositing, no depth test) | Just added to the main renderer |

**The scalar-bar duplication problem** is real for `results.viewer`. Today, two Contour
diagrams coloring by the same array get two scalar bars (or fight for the same screen
position). Plan 06 plans a LUT manager — it should also deduplicate scalar bars. **Plan
06 needs an explicit update for this** (see below).

The grid axes (X/Y/Z gridlines with numeric labels) and the time annotation overlay are
*free wins*. PyVista exposes `plotter.show_grid()` and `plotter.add_text(...)`; a one-day
overlay module would add both, themed properly. Worth listing as a small follow-up.

---

### 5. Camera and lighting

| Aspect | ParaView | apeGmsh |
|---|---|---|
| Camera | Shared by composited + non-composited renderers; reset via `ResetCameraScreenSpace` | Single PyVista camera; reset via `plotter.reset_camera()` ([viewer_window.py:471-476](../../src/apeGmsh/viewers/ui/viewer_window.py)) |
| Lighting | `vtkLightKit` — three-light kit (key, fill, back), tunable warmth and elevation | PyVista's default single headlight |
| Background | Solid, gradient, image, skybox, environment map | Themed gradient via `scene.background.apply_background` ([viewer_window.py:547](../../src/apeGmsh/viewers/ui/viewer_window.py)) |
| Parallel projection | `vtkCamera::ParallelProjection` | `enable_parallel_projection()` ([viewer_window.py:268-271](../../src/apeGmsh/viewers/ui/viewer_window.py)) — defaults on |

**Mostly fine for us.** A three-light kit gives slightly better-looking shaded surfaces
on geometry — useful for `model.viewer`, marginal for results. PyVista's default
headlight is OK. Skybox / environment map is overkill.

One small gap: parallel projection is on by default in our viewers (correct for
engineering use), but ParaView lets the user toggle per-view. We have the toggle in the
toolbar ([viewer_window.py:268-271](../../src/apeGmsh/viewers/ui/viewer_window.py)) — no change needed.

---

### 6. Selection rendering

| Aspect | ParaView | apeGmsh |
|---|---|---|
| Mechanism | `vtkPVHardwareSelector` — separate render pass with color-coded IDs, pixel readback maps to data IDs | Re-color via `ColorManager` in the same actor |
| Selection visual | Highlighted overlay (separate rep) | Re-colored cells / nodes in the original actor |
| Through transparency | Works (separate pass) | Doesn't — selection is hidden behind transparent contour overlays |

**Where this hurts:** in `results.viewer`, picking an element through a transparent
contour layer is impossible — the re-colored element is occluded by the transparent
mesh. ParaView's overlay approach avoids this.

For our scale, this is a known-bug-class but not a blocker. Filing for `future/`.

---

## Six rendering-layer wins, ranked

| # | Win | Cost | Where it lands |
|---|---|---|---|
| 1 | **Depth peeling enabled on the renderer** (fixes z-fighting in overlapping transparent contour/section diagrams) | 4h | Should be folded into plan **06** (color-map editor — since transparency settings live there) |
| 2 | **Scalar bar deduplication** (LUT manager owns one bar per LUT; reps reference it) | 3h | **Update plan 06** to include this in the LUT manager API |
| 3 | **Still vs interactive render**: hook mouse press/release, drop quality on drag | 6h | Standalone — small follow-up after the must-ship list |
| 4 | **Grid axes overlay** (`plotter.show_grid()` themed properly) + **time-text overlay** | 1d | Standalone — small follow-up |
| 5 | **Selection rendered as overlay** (separate actor on top, ignoring transparency) | 2d | `future/selection-overlay.md` — wait for selection-system refactor |
| 6 | **Representation abstraction split** (data extraction ↔ rendering decoupled) | 1–2 weeks | `future/representation-split.md` — the big one, defer |

Items 1–4 add up to about ~2.5 days of work and would noticeably improve `results.viewer`.

---

## What this means for the existing plan docs

Three concrete adjustments:

### A. Update [`06-color-map-editor.md`](06-color-map-editor.md)

Add two items to the LUT manager API:

- **`LUT` carries its scalar bar.** Today each Diagram creates its own. The LUT manager
  hands out a scalar bar reference along with the LUT; reps that share a LUT share the
  bar. Toggling "Use Separate Colormap" gives both a separate LUT *and* a separate bar.
- **Depth peeling toggle in the editor's "Render quality" section**, tied to the renderer
  globally — it's not per-LUT, but the editor is where users discover it.

### B. Demote [`02-view-frame-chrome.md`](02-view-frame-chrome.md)

The toolbar in [`viewer_window.py:252-298`](../../src/apeGmsh/viewers/ui/viewer_window.py) already has reset, fit, axis snaps,
parallel-projection toggle, and screenshot — that's most of the plan 02 work. What's
missing is the *extensibility hook* (`view_frame.add_action(QAction)`) for diagrams and
overlays to register their own buttons. Plan 02 should be retitled to "Toolbar
extensibility hook" and shrunk to ~0.5 day. The "discoverability" framing is wrong —
discoverable buttons already exist.

### C. Stub two new `future/` docs

- `future/render-quality.md` — depth peeling, still/interactive, LOD if ever needed.
- `future/representation-split.md` — the data ↔ render decoupling of `Diagram`.
- `future/selection-overlay.md` — selection as a separate render layer.

I'll create the stubs if you want; they're empty placeholders that map the deferred
work, not full plans yet.

---

## What we should NOT port

- **OSPRay / ANARI ray tracing.** ParaView wires these in; we don't need photorealistic
  rendering.
- **FXAA.** PyVista's `enable_anti_aliasing(...)` is enough and is already wired
  ([viewer_window.py:174-178](../../src/apeGmsh/viewers/ui/viewer_window.py)).
- **vtkPVLODActor.** Full LOD geometry swap is overkill for our typical model sizes
  (<100k cells). Still vs interactive *without* a real LOD geometry is enough.
- **Multi-process render decomposition (IceT).** Single-process only.
- **Skybox / environment map backgrounds.** Themed gradient is fine.
- **Three-light kit.** PyVista's default headlight is OK; engineering visualization
  doesn't need a Hollywood lighting rig.

---

## Honest summary

The plotting layer in apeGmsh is not broken — it's just thin. PyVista does a lot for us
for free. The two things genuinely missing are (1) depth peeling for transparency and
(2) scalar-bar / LUT deduplication. Both fold into the work that's already planned (plan
06). Everything else is either a small follow-up (still/interactive, grid axes, time
overlay) or a long-term architectural win (representation split) that belongs in
`future/`.

If we're being disciplined: the plotting comparison adds ~1 day to plan 06 and one new
small follow-up doc, not a new must-ship item.
