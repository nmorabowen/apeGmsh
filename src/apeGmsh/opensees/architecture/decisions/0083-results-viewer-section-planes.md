# ADR 0083 — Section planes: viewer-owned clipping across the render seam

**Status:** Proposed (2026-07-27) — design ratified by the user
2026-07-27 (three scope forks resolved: hollow-first, panel-first,
desktop-first). Flips to Accepted at close-out with per-slice PR
numbers, per house convention.

Lands under ADR 0056 (viewer state & event contract): plane state gets
exactly one owner, a viewer-level controller, and every mutation fires
one dispatcher event. Constrained by ADR 0042 (the cut must cross the
render seam — diagram/domain code never imports VTK). Follows the
ADR 0081 pattern twice over: a view annotation owned by the viewer,
and — in slice 2 — the priority-12 own-event-layer interactor that
0081 L2 proved out.

## Context

The results viewer needs SketchUp-style section planes: create one or
more planes, drag them through the model, and see the interior of the
contoured solid — including while scrubbing time on the deformed
shape. Today the results viewer has **no clipping of any kind**; a
grep for `clip` across `scene_ir/`, `diagrams/`, `backends/` finds
only `np.clip`, camera near/far planes, and section-cut docstrings.

### What exists elsewhere, and why it cannot be reused

- The mesh viewer's [`ClippingController`](../../../viewers/core/clipping_controller.py)
  (single `vtkPlane`, PyVista `add_plane_widget`) and the model
  viewer's [`ClipPlaneOverlay`](../../../viewers/overlays/clip_plane_overlay.py)
  + [`ClipPlanePanel`](../../../viewers/ui/_clip_plane_panel.py)
  (axis-aligned, slider-driven) both walk registry actors and call
  `mapper.AddClippingPlane` directly. The results viewer renders
  through the `RenderBackend` Protocol
  ([`scene_ir/_backend.py:42`](../../../viewers/scene_ir/_backend.py))
  — INV-2 forbids that direct mapper access, so neither implementation
  crosses over. They stay as-is; this ADR does not touch them.
- **The name "section cut" is taken.** `SectionCutDef`
  ([`cuts/_defs.py:76`](../../../cuts/_defs.py)) is the
  force-integration feature, rendered by `SectionCutDiagram`
  (`kind="section_cut"`, composition name "Section cuts"). This
  feature is **"Section planes"** in the UI and `clip_plane` in code
  (`dock_results_clip_planes`, `ClipPlaneSetController`, session field
  `clip_planes`) — four existing names avoided.

### The seam facts that shape the design

- One backend instance per viewer session:
  `DiagramRegistry.bind` wraps the plotter in `PyVistaQtBackend` once
  ([`diagrams/_registry.py:110`](../../../viewers/diagrams/_registry.py))
  and every diagram attaches to it. `TrameBackend` subclasses
  `PyVistaBackend` unchanged (server-side VTK), so a backend-level
  clipping API is inherited by the web viewer.
- The backend does **not** track the handles it issues
  ([`pyvista_qt.py:264`](../../../viewers/backends/pyvista_qt.py)) —
  there is no central iterable of live mappers today.
- `update_layer`'s rebuild path destroys and re-creates the actor
  ([`pyvista_qt.py:327-334`](../../../viewers/backends/pyvista_qt.py))
  — every `GlyphLayer` update and every topology change gets a fresh
  mapper. Any clipping state attached to the old mapper dies there.
- Substrate fill / wireframe / node cloud are **off the seam**: the
  viewer adds them straight to the plotter, at boot
  ([`results_viewer.py:944`](../../../viewers/results_viewer.py)) and
  again per materialized geometry
  ([`results_viewer.py:1618`](../../../viewers/results_viewer.py)).
- Deformation rewrites real point coordinates every step
  (`_pump_deform`, [`results_viewer.py:1355`](../../../viewers/results_viewer.py));
  actors are in true model coordinates (no `origin_shift` in the
  results viewer). A world-fixed plane therefore cuts the *rendered*
  (deformed) shape — the wanted behaviour.
- Picking is a `vtkCellPicker` ray-cast
  ([`backends/_pyvista_pick.py:223`](../../../viewers/backends/_pyvista_pick.py))
  — a geometric intersection with the dataset. Mapper clipping planes
  are a rasterization-time effect the picker knows nothing about.
- The pick engine aborts every plain LMB press at VTK priority 10
  ([`_pyvista_pick.py:185`](../../../viewers/backends/_pyvista_pick.py)),
  so a stock `add_plane_widget` (priority 0.5) would never receive a
  click — the same failure ADR 0081 measured for `vtkScalarBarWidget`.

### Ratified scope forks

- **(F1) Hollow first.** Slice 1 uses render-time clipping
  (`AddClippingPlane`): fragments on the cut side are discarded, the
  interior of solids reads as a hollow shell (still contoured).
  Interpolating the scalar field onto the flat cut face is a dataset
  pass deferred to slice 3.
- **(F2) Panel first.** Slice 1 is dock-panel-driven (axis chips,
  offset slider, numeric entry). The draggable in-viewport gizmo is
  slice 2, because it requires the priority-12 interactor treatment.
- **(F3) Desktop first.** The backend API lands on the Protocol so
  trame inherits it; no web UI controls in slice 1.

## Decision

Five parts. Parts 1–2 put the state and the cut in the right places,
Part 3 covers the off-seam actors, Part 4 is the GUI, Part 5 makes it
durable and provable.

### Part 1 — Plane state has one owner: `ClipPlaneSetController`

A new controller in `viewers/core/`, one per viewer, sibling of
`LegendController`:

```python
@dataclass
class ClipPlane:
    plane_id: str
    name: str                     # "Plane 1", user-renamable later
    normal: tuple[float, float, float]   # unit, world-space
    offset: float                 # signed distance origin→plane along normal
    active: bool = True           # is it cutting
    flipped: bool = False         # which half survives (SketchUp "Reverse")
    gizmo_visible: bool = True    # slice 2
```

Plus two set-level booleans mirroring SketchUp's display toggles:
`apply_cuts` (master enable) and `show_gizmos`. Planes are
**world-space by definition** and cut whatever is currently rendered —
the deformed shape moves through them as the scrubber plays. The
per-geometry `offset` feature (ADR 0058 S3a) slides a geometry
*through* a world-fixed plane; that is documented behaviour, not a
bug (see Open question 1).

At most **6 planes may be active** at once — the OpenGL guaranteed
minimum for user clip distances, which VTK inherits. The controller
enforces it (activating a 7th is refused with a status message);
creating more than 6 *inactive* planes is fine.

Per ADR 0056: mutators live on the controller, each fires exactly one
`CLIP_PLANES_CHANGED` dispatcher event with a render-only primitive
row. The panel and (slice 2) the gizmo are projections.

### Part 2 — The cut crosses the seam as backend-owned stamping state

`RenderBackend` widens by one method:

```python
def set_clip_planes(self, planes: Sequence[ClipPlaneSpec]) -> None:
    """Replace the scene's active clip set. Empty sequence = no cut.

    Applies to every live layer AND every layer added afterwards —
    the backend stamps its current clip set onto each actor it
    creates. Non-3D layers (labels, 2D annotations) are exempt.
    """
```

`ClipPlaneSpec` is a frozen scene-IR value type (origin, normal) —
the controller resolves `offset`/`flipped` into it, so the backend
sees only ready-to-use half-space definitions.

Implementation in `PyVistaBackend`:

- The backend gains the handle registry it lacks: `_add_mesh_layer` /
  `_add_glyph_layer` record their `_PvHandle`s (weakly, keyed by
  `layer_id`; `remove_layer` drops the entry).
- `set_clip_planes` rebuilds an internal `vtkPlaneCollection`-worth of
  `vtkPlane`s and applies it to every registered mesh/glyph mapper
  (`RemoveAllClippingPlanes` + `AddClippingPlane` per plane).
- **Stamping, not attaching**: `_add_mesh_layer` / `_add_glyph_layer`
  apply the current clip set to each new mapper at creation. This is
  what makes the `update_layer` rebuild path
  ([`pyvista_qt.py:327`](../../../viewers/backends/pyvista_qt.py))
  safe — the re-created actor is stamped on its way in, so a glyph
  diagram animating through a cut stays cut. The in-place fast path
  never touches the mapper, so it is safe by construction.
- `LabelLayer` handles are never stamped (`kind` check) — 2D text
  mappers do not support world clip planes and labels should remain
  readable regardless of the cut.

Because the planes live in backend state, `TrameBackend` inherits the
whole mechanism (F3). One scope note: the inheritance claim is proven
for **server-side rendering**; a client-side vtk.js render mode would
need the planes serialized into the client scene, which is untested —
see Open question 2.

### Part 3 — Off-seam actors are stamped at their creation sites

Substrate fill, wireframe, and the node cloud bypass the backend, so
the viewer attaches the controller's planes itself — **inside**
`_add_substrate_actors` and the node-cloud builder, not after them.
Attaching at the creation helper covers both call sites (boot at
`:944`, per-geometry materialization at `:1618`) and any future one,
the same reasoning that puts backend stamping inside `add_layer`. The
`CLIP_PLANES_CHANGED` handler re-applies to all currently live
off-seam actors.

A clipped node-cloud sphere renders as a half-sphere at the cut —
acceptable; excluding whole nodes by center would misreport which
nodes exist near the plane.

### Part 4 — GUI

- **Dock panel** — extension dock via `DockSpec`
  (`dock_id="dock_results_clip_planes"`, title "Section planes",
  `tabify_with="dock_results_diagram"`), registered exactly like the
  Color-map editor and Definitions docks
  ([`results_viewer.py:640-662`](../../../viewers/results_viewer.py)).
  Bump `_LAYOUT_SCHEMA_VERSION` (5 → 6).
  Contents (mockup ratified 2026-07-27): *Add plane* button with an
  orientation dropdown (X / Y / Z / current view normal); one row per
  plane with active-checkbox, flip, gizmo-eye (slice 2), delete;
  orientation chips X/Y/Z/Custom (Custom expands three normal
  fields); an offset slider + numeric field. The slider maps the
  **reference-geometry bbox** along the plane normal; the numeric
  field is unclamped, because deformation scale and per-geometry
  offsets legitimately push geometry outside the reference range.
  Footer: the two global toggles (*Apply cuts*, *Show plane gizmos*)
  and a disabled *Contour on cut face* placeholder pointing at
  slice 3.
- **Toolbar** — one checkable action via
  `ResultsWindow.add_toolbar_action` (pattern:
  [`results_viewer.py:705`](../../../viewers/results_viewer.py)) that
  raises the dock and toggles `show_gizmos`.
- **Shortcuts-help entry** added to the F1 dialog list. Real key
  bindings (if any) follow the `ApplicationShortcut` rule — the VTK
  interactor swallows window-scope keys.
- **Gizmo (slice 2)** — a translucent quad + normal arrow per
  visible plane; drag along the normal, rotate via handle. It does
  **not** use `add_plane_widget`: it installs its own observers at
  priority **12** and aborts only when the press hits a gizmo handle
  — the exact `LegendInteractor` pattern ADR 0081 L2 shipped and
  measured. Miss → no abort → picking and navigation bit-for-bit
  unchanged (0081's L-PICK guard extends to cover it).

### Part 5 — Durable and provable

**Persistence.** `ViewerSession` gains a top-level
`clip_planes: tuple[ClipPlaneSnapshot, ...] = ()` plus the two set
booleans — the `legends` pattern from schema v8
([`diagrams/_session.py:121`](../../../viewers/diagrams/_session.py)).
`SESSION_SCHEMA_VERSION` 8 → 9; a missing field reads as "no planes",
so old sessions load clean. Restore applies planes to the controller
after `bind_plotter` (the backend must exist); ordering relative to
diagram attach is immaterial because stamping covers late layers.

**Pick coherence.** The `vtkCellPicker` would happily return a cell
the cut has hidden — a phantom pick on empty-looking space. The
results pick controller gains a post-pick filter: if the hit's world
point is on the discarded side of any active plane, the pick is
rejected (treated as a miss). Picking *through* the hole onto visible
geometry behind it is **not** attempted in slice 1 — a documented
limitation (see Open question 3). The navigation orbit-anchor picker
([`core/navigation.py:391`](../../../viewers/core/navigation.py)) can
likewise anchor on clipped geometry; cosmetic, accepted.

**Tests assert the picture** (the 0081 lesson: verify pixels and
behaviour, not that VTK was called):

- **C-CUT** — offscreen render with a plane through a contoured brick:
  pixel counts prove geometry is gone on the cut side and present on
  the kept side; flip inverts it.
- **C-REBUILD** — a glyph diagram steps (rebuild path) while a cut is
  active: the re-created actor's mapper carries the planes. This is
  probe P1's regression test.
- **C-SUBSTRATE** — a second geometry materializes with a cut active:
  its substrate actors arrive clipped (probe P2).
- **C-PICK** — a click on a clipped-away cell reports a miss; the
  same click with cuts off reports the cell (probe P3).
- **C-DEFORM** — with deformation on, the set of clipped points
  changes between step 0 and step N for a fixed plane (the cut tracks
  the warp).
- **C-SESSION** — round-trip: two planes (one active, one flipped)
  survive save/load; a v8 session loads with zero planes.
- **C-EXPORT** — `export_animation` frames show the cut (probe P6's
  pin).
- **C-LIMIT** — activating a 7th plane is refused; six work.

## Design-review probes (pre-implementation adversarial pass)

Eight probes against the first draft of this design; three were live
defects in the draft, each now folded into the Decision above.

| # | Probe | Finding | Disposition |
|---|---|---|---|
| P1 | Does clipping survive `update_layer`? | **Defect.** The draft attached planes to live mappers once; the rebuild path (`pyvista_qt.py:327-334`) re-creates actor+mapper on every glyph update and topology change — the cut would silently vanish from animating diagrams. | Redesigned as backend **stamping** state applied inside `add_layer` (Part 2). C-REBUILD guards it. |
| P2 | Are substrate actors covered after boot? | **Defect.** The draft attached planes to the boot-time substrate only; `_add_substrate_actors` runs again per materialized geometry (`results_viewer.py:1618`) — a geometry added later would render uncut. | Attach moved **inside** the creation helpers (Part 3). C-SUBSTRATE guards it. |
| P3 | Does picking respect the cut? | **Defect.** `vtkCellPicker` is a geometric ray-cast (`_pyvista_pick.py:223`); mapper clip planes are invisible to it — clicking apparently-empty space would pick hidden cells and drive the probe HUD with phantom data. | Post-pick clipped-side rejection filter (Part 5). Pick-through deferred (Open question 3). C-PICK guards it. |
| P4 | Slider range vs. deformed/offset geometry | Slider mapped to the reference bbox can't reach geometry pushed outside it by deform scale or per-geometry offset. | Not a defect of the cut itself; resolved by the unclamped numeric field (Part 4). |
| P5 | How many planes can a mapper take? | OpenGL guarantees only 6 user clip distances. | Controller enforces ≤ 6 active (Part 1). C-LIMIT guards it. |
| P6 | Does video export lose the cut? | **Non-finding.** `export_animation` reuses the live plotter ([`animation.py:36-51`](../../../viewers/animation.py)) — frames inherit whatever the mappers carry, including planes (given P1's fix). | Pinned by C-EXPORT rather than assumed. |
| P7 | Does the trame claim hold in every render mode? | Server-side rendering inherits mapper state trivially; a client-side vtk.js path would need plane serialization — unverified. | Claim scoped to server-side rendering (Part 2); Open question 2. |
| P8 | What happens to label/2D layers? | `AddClippingPlane` on 2D text mappers is at best a no-op, at worst an error; labels near the plane would half-vanish confusingly if it worked. | Backend stamps mesh/glyph kinds only (Part 2). |

## S1 adversarial review (post-implementation, 2026-07-27)

S1 shipped as six commits (`251c2c85`…`c8cbc209`). The review probed
the implementation live — scripts against the worktree, not re-reads.
One live defect, fixed with a regression test; the sign-convention
risk was verified closed; the implementer's own flagged deviations
were dispositioned.

| # | Finding | Disposition |
|---|---|---|
| A1 | **Duplicate plane ids after a sparse restore.** `restore()` reset the id counter to `len+1`; a session saved after a deletion restores sparse ids (plane-1, plane-3) and the next `add()` minted plane-3 again — measured: `set_offset` on the new plane mutated the *restored* plane, so every id-keyed panel action drove the wrong row. | Fixed (`70498631`) — counter starts past the highest numeric suffix among restored ids; `test_a_plane_added_after_a_sparse_restore_gets_a_fresh_id`. |
| A2 | Render side vs. `is_clipped` side could disagree (two consumers of one convention). | **Not a defect** — both consume the same `ClipPlane.spec()`, and C-CUT pixel-pins which half survives for both normal directions, so a divergence cannot pass the suite. |
| A3 | Toolbar action raises the dock only, does not toggle `show_gizmos` (implementer deviation from one Part-4 sentence). | Accepted for S1 — a toolbar toggle would fight the footer checkbox, and gizmos are inert until S2. S2 revisits. **Premise refuted in Round 2 (B1): the action did not even open the dock when tabified behind Diagram — see below.** |
| A4 | Weak handle registry: a live actor whose handle the domain layer dropped stops being re-cut (keeps its stale stamp). | Accepted — diagrams hold their handles today; noted as an S2 reviewer checkpoint if any new layer owner appears. |
| A5 | The live `show()` path (dock mount, toolbar, panel binding, session restore in a real window) has no automated coverage — `pytest-qt` absent from the venv. | Open — one manual viewer open owed before the PR merges. |
| A6 | Sand / fiber point clouds: distinct code path claim. | **Covered by construction** — they are vertex-cell `MeshLayer`s through `_add_mesh_layer` (stamped) and their per-step update is the in-place fast path (mapper untouched). |

### Round 2 (fresh-eyes pass, 2026-07-28)

A second, independently-contexted review probed what round 1 accepted
on reading: the viewer wiring and the Qt panel. Three confirmed
defects and one UX defect, all fixed with regression tests
(`test_clip_planes.py`, B1–B4 block); round 1's A3 disposition was
**refuted** in the process.

| # | Finding | Disposition |
|---|---|---|
| B1 | **The toolbar button was dead whenever the Diagram dock was visible.** The wiring copied the colour-map pair (`toggled → setVisible`, `visibilityChanged → setChecked`) — but this dock is *tabified*, and Qt reports a background tab as not-visible, so the click that opened the dock immediately closed it. Measured with the real dock + real mounting. Refutes A3's premise ("raises the dock only") — it did not open it at all. | Fixed — `wire_tabified_dock_action` in `ui/_dock_registry.py`: open = `show()`+`raise_()`; check-state written back from `visibilityChanged` is guarded and never drives visibility. `test_toolbar_action_fronts_a_tabified_dock`. |
| B2 | **A planes-only session was saved but never restored.** `_maybe_restore_session` gated on `if not session.diagrams: return`; planes are the first diagram-independent session payload, so cutting the bare substrate, saving, and reopening silently dropped the planes. | Fixed — gate now `session_restorable(session)` (in `diagrams/_session.py`): diagrams **or** clip planes. `test_a_planes_only_session_is_restorable`. |
| B3 | **Typing decimal offsets / custom normals was impossible.** Per-keystroke `valueChanged` fired `CLIP_PLANES_CHANGED`, whose UI-lane `refresh()` rewrote the focused editor between keystrokes — measured: typing `2.5` committed 2.0, and a custom normal could not be entered at all; the Custom chip also collapsed on any unrelated refresh (expansion was derived from the normal). | Fixed — `setKeyboardTracking(False)` on all four numeric fields, `_sync_selected` never rewrites a focused editor or a held slider, and "Custom expanded" is panel state. `test_numeric_fields_commit_on_enter_not_per_keystroke`, `test_custom_expansion_survives_unrelated_refreshes`. |
| B4 | **Add plane at world offset 0 could discard the whole model** (bbox not straddling the origin — georeferenced coordinates). | Fixed — a new plane lands at the centre of the bbox span along its normal. `test_added_plane_lands_mid_geometry`. |

Round 2 also probed clean: render-after-re-cut ordering (RENDER-lane
subs run before the dispatcher's paint; the `_apply_session` batch
compensates explicitly post-batch), slider-drag coalescing, reentry on
the refused 7th checkbox, `_offset_range` for oblique normals, session
deserialize edge cases (`null` / missing fields → skip-or-default),
INV-2, Trame inheritance, subscription lifetimes (per-viewer
dispatcher — not the 0081 global-singleton leak class), and the v5→v6
layout-state discard. A5 (one manual live-window open) remains owed
and now explicitly includes the B1 case: check the toolbar button
**with the Diagram dock visible**.

## Alternatives considered and rejected

1. **Reuse `ClipPlaneOverlay` / `ClippingController` directly.**
   Both violate INV-2 in the results viewer (direct mapper access),
   support exactly one plane, and know nothing about layers created
   after enable — which is the common case here (P1/P2). The
   model-viewer panel UX is inherited; the mechanism is not.
2. **True dataset clipping in slice 1** (`vtkTableBasedClipDataSet`
   per layer per step). Correct cut faces, but it recomputes geometry
   on every drag tick and every animation step of every layer —
   exactly the cost render-time clipping exists to avoid, and it
   fights the `update_layer` fast path. Deferred to slice 3 as an
   *additional* cap pass on drag-release, not a replacement.
3. **Per-diagram clip property** (each diagram clips its own layers).
   The 0081 root-cause shape: N owners for one piece of view state,
   and the substrate/node cloud have no diagram to own them. The cut
   is a property of the *view*, so it is owned by a viewer-level
   controller.
4. **Stock `add_plane_widget` for the gizmo.** Dead on arrival at
   priority 0.5 under the pick engine's unconditional abort at 10 —
   the measured 0081 D1 failure. The own-event-layer interactor at 12
   is proven and keeps the gizmo's look consistent with the house
   style.

## Consequences

- `RenderBackend` widens by one method; backends gain a handle
  registry they arguably should have had already (0081's Part 3 also
  wanted a central layer iterable). Trame inherits the cut for
  server-side rendering with no web-specific code.
- The viewer owns one more controller + dispatcher kind, per the
  ADR 0056 pattern — no new state homes anywhere else.
- Render-time clipping means solids look hollow at the cut in slices
  1–2. Accepted (F1); slice 3 exists because of it.
- **Risk:** stamping inside `add_layer` couples layer creation to
  clip state — a backend bug there would clip layers that should be
  exempt. Mitigated by the kind check (P8) and C-CUT/C-REBUILD.
- **Risk (slice 2):** another priority-12 interactor alongside the
  legend's. Both abort only on their own hit-test, so they compose;
  the gizmo installs *after* the legend interactor so legends win a
  literal overlap (a gizmo edge behind a legend box).

## Runway

- **S1 — panel-driven planes.** `ClipPlaneSetController` +
  `CLIP_PLANES_CHANGED`; `RenderBackend.set_clip_planes` + handle
  registry + stamping; off-seam attachment; dock panel + toolbar
  action + shortcuts-help entry; session v9; pick filter; tests
  C-CUT through C-LIMIT. Fully usable without a gizmo.
- **S2 — the gizmo.** Priority-12 interactor (0081 L2 pattern), quad
  + normal-arrow rendering as scene-IR layers (exempt from stamping),
  drag-along-normal, rotation handle, per-plane gizmo visibility,
  "align to view" placement.
- **S3 — contour on the cut face (optional).** Dataset slice at each
  active plane on drag-release / step change, interpolating the
  active scalar onto the cut polygon; rendered as an ordinary
  scene-IR mesh layer exempt from stamping.

## Open questions

1. **Per-geometry offset semantics.** A world-fixed plane cuts offset
   geometries at a different model-relative station. Proposal: keep
   world-space planes and document it; revisit per-geometry plane
   binding only if side-by-side comparison workflows actually demand
   it.
2. **Client-side web rendering.** If `show_web` is ever run in a
   vtk.js client-render mode, do serialized scenes carry mapper clip
   planes? To be measured when (if) web controls ship; until then the
   web claim is server-side only.
3. **Pick-through.** Should a click on the cut hole fall through to
   the first *visible* cell behind it (requires walking multiple ray
   hits) instead of reporting a miss? Proposal: ship the miss
   behaviour, collect real usage, promote pick-through to a slice-2
   task only if the miss feels wrong in practice.
