# ADR 0098 Amendment 7 (R1) — design brief: the clip gizmo comes back, the scope gizmo does not

Status: design, for the owner. Not an ADR. If accepted, A7 is written from
this and the parity registry moves in the same commit.

Scope: A6.6's R1. Amendments 1–6, §1–§11, INV-MESH-1…4, INV-LEGEND-1…5 all
stand. **This one reaches the IR only in the reconciler's signature split** —
no session record gains a field.

---

## 1. Context

A6.6 retracted A5.5: R1 is not a wiring job. The session emits no gizmo
actors, so there is nothing for `install_clip_gizmo_interactor` to grab.
This brief settles the five questions that stand between that finding and
an implementable slice plan, and it corrects three things the working brief
had wrong.

Slice 1 has already landed in the worktree and is **unused and untested**:
`src/apeGmsh/viewers/session/_clip_control.py::ViewClipController`, the
five-member ADR 0083 controller contract served off a `MeshView`. This brief
ratifies it (Q1, Q2) and names what has to be built on top.

---

## 2. Evidence

### 2.1 The contract is five members and `ViewClip` already fits it

Extracted from both consumers, not from the old controller's surface:

| member | consumer |
|---|---|
| `planes()` | `_clip_gizmo.ClipGizmoRenderer.refresh` |
| `show_gizmos` | same |
| `plane(plane_id)` | `_clip_gizmo_interactor._on_lmb_press` |
| `set_offset(id, offset)` | translate drag |
| `set_pose(id, normal, offset)` | rotate drag |

The plane objects must expose `plane_id`, `normal`, `offset`, `flipped`,
`gizmo_visible`. `ViewClip` (`results/session/_views.py:171`) has all five,
identical names. `ClipGizmoRenderer` and `install_clip_gizmo_interactor`
both exist and work; every construction site is in `results_viewer.py`
(2213, 4336). The session builds none.

### 2.2 The drag frame costs 240.8 ms — measured, not feared

`internal_docs/profiling/results_session/profile_session.py`, gesture
`clipdrag` (one `view.set_clip(plane_id, offset=…)` per iteration = one
mouse-move of a translate drag), bench `ssi_frame_wall`, case
`c010_clip_gizmo_drag` (24 PASS / 0 FAIL), `--panes 1 --repeat 20
--no-profile`:

| gesture | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| **clipdrag** | 20 | **240.8** | 233.7 | 252.2 | 283.5 |
| scrub | 20 | 17.7 | 16.7 | 21.5 | 30.3 |
| orbit | 20 | 19.8 | 16.7 | 34.8 | 62.7 |

Budget is the scrubber's 33 ms (A4's own). **7.3x over — about 4 fps for a
gesture whose entire purpose is direct manipulation.** Scrub and orbit are
the controls and they matter: A4 took scrub from 405 to 17.7 ms, and orbit
at 19.8 ms is the pure-GL floor, so the machine and the pane are fine.
Clipdrag is 13.6x a scrub on the same pane in the same process. **The clip
drag is now the most expensive gesture in the session window** — dearer
than the one A4 was written to fix, and worse in kind: a scrub frame is a
frame the user asked for, a drag frame is one of dozens inside a single
gesture.

### 2.3 Why A4's fast path does not and cannot help

`pane.clips` sits in the **structure** half of `_pane_signature`
(`_reconciler.py:338`), beside `pane.slots` and `pane.legends()`. The
session tick is synchronous (`_session.py:352` iterates subscribers inline;
`MeshView._changed` calls `self._notify()` directly). So one mouse-move is:

    interactor.set_offset → ViewClipController._set → view.set_clip
      → _changed → session._tick → reconciler._flush
      → structure differs → _teardown + realize_pane (every layer, every bar)

`restep_pane` fires only when the **cursor** term moved alone, so
`can_restep` is never consulted. The 240.8 ms is a full teardown-and-rebuild
per frame.

This is exactly the trap A5.3 dodged for legends: it refused to put
placement on the `Legend` record precisely because `Legend` is in the
structure signature and it would cost "about 150 ms per frame on the bench".
**Clips are already there.**

### 2.4 A clip-only change needs three calls, and no realize

`_apply_clips` (`_realize.py:1542`) is the only place realize reads
`view.clips`, and `PyVistaQtBackend.set_clip_planes`
(`backends/pyvista_qt.py:754`) **re-stamps every live handle**, not only
new ones. So the complete effect of a clip edit on an already-realized pane
is: `set_clip_planes(specs)`, `gizmos.refresh()`, `render()`. Nothing else
in realize is a function of `view.clips`.

### 2.5 Two defects found while writing this (A6.1 style)

* **The six-plane GL cap is not enforced on the session path.**
  `ClipPlaneSetController` refuses a seventh active plane out loud
  (`MAX_ACTIVE_PLANES`, `_refuse` — `_clip_planes.py:47,338`), because past
  six the driver is free to ignore a plane. `_apply_clips` pushes whatever
  is active. The session can therefore activate a seventh plane and get **a
  cut that silently lies**. Not R1's job; recorded so it is not
  rediscovered.
* **Pick coherence is not ported.** ADR 0083 Part 5's `is_clipped` filter
  keeps `vtkCellPicker` — a geometric ray-cast that knows nothing about
  mapper clip planes — from returning a cell the cut is hiding. Grep:
  no `is_clipped` anywhere under `viewers/session/`. A click on apparently
  empty space in front of a cut returns the hidden cell. Also not R1's job,
  but R1 makes clipping usable, which makes this reachable.

### 2.6 No UI creates a clip

There is no clip row in `_outline.py`, no clip context in the inspector, and
`make_clip_planes_dock` has one caller: `results_viewer.py:975`. On the
session path a clip exists only if the user typed `view.add_clip(...)` or
the run carried persisted cuts (`results/session/_cuts.py`). **A gizmo for
planes that cannot be created is the A6.1 row "works from Python /
unreachable in the UI" in a new costume.** Reach is therefore in R1 (§4.4),
not deferred.

---

## 3. Decision table

| Q | Decision | Serves |
|---|---|---|
| Q1 `show_gizmos` | **Derived**: `any(clip.gizmo_visible)`. No new field, no viewer-wide flag. Ratifies the landed `ViewClipController.show_gizmos`. | A5.3's "existence stays derived"; §3's single ownership |
| Q2 renderer owner | **Pane-owned** `PaneClipBinding` (`session/_clip_bind.py`), mirroring `PaneLegendBinding`. Long-lived renderer, long-lived interactor, layers added through the **raw** backend, never the `LedgerBackend`. | A5.2's stale-controller measurement; `_clip_gizmo.py:376`'s strong-ref contract |
| Q3 two panes, one view | **Dissolved.** No such case exists. | Correction B, §3 |
| Q4 drag-frame cost | **(a)** — `clips` becomes its own signature term and gets a restep-style fast path (`reclip_pane`: `_apply_clips` + `gizmos.refresh()` + the existing single `render()`). No teardown, no `realize_pane`. | A4.2's shape, applied to a second term |
| Q5 scope gizmo | **`retired`**, with a `why`. It is not the same shape as the clip gizmo. | Correction C; A2.2's disposition precedent |

### Q1 — where `show_gizmos` lives

Derived, and the derivation is exact rather than approximate. The renderer
draws plane `P` iff `show_gizmos and P.gizmo_visible`
(`_clip_gizmo.py:396-401`); with `show_gizmos = any(P.gizmo_visible)` that
reduces to `P.gizmo_visible` for every `P`, for every combination of flags.

* **The user gesture.** "Hide all gizmos" maps onto *clear every plane's
  `gizmo_visible`*, one edit per plane in one tick. There is no separate
  master state to persist.
* **Snapshot.** Nothing new. `gizmo_visible` already round-trips
  (`_snapshot.py:239`, `:475`). A stored viewer-wide flag would need its own
  field, its own restore rule and its own "what if it disagrees with the
  per-plane flags" answer.
* **The cost, stated.** A master toggle written this way is **destructive**:
  hide-all then show-all returns every plane to visible and forgets which
  ones the user had individually hidden. R1 therefore ships **no master
  toggle** — with one to three planes the per-plane eye is the whole
  control, and a master switch over one plane is a control that acts on
  nothing (0087 INV-2). If a master toggle is later wanted for a
  many-plane view, the honest form is a new *chrome* field on `MeshView`
  (`gizmos_hidden`, sibling of `legend_hidden`) and it is an amendment, not
  an implementation detail.

### Q2 — who owns `ClipGizmoRenderer`

`PaneClipBinding(backend, reconciler)`, constructed in `MeshPane.__init__`
beside `PaneLegendBinding` (`session/_pane.py:202`), owning:

1. one `ViewClipController` per bound view,
2. one `ClipGizmoRenderer(backend, controller, bbox=…)`,
3. one `ClipGizmoInteractor` (from `install_clip_gizmo_interactor`).

Four mechanics decide this:

* **The ledger would eat a realize-owned renderer's handles.** Realize
  emits through `LedgerBackend`, and `_teardown` removes every handle in it
  (`_reconciler.py:372-378`). Gizmo layers added through the **raw** backend
  are outside the ledger, so a realize cannot sweep them, and the renderer's
  strong refs (`_clip_gizmo.py:376`) stay valid. The binding's `dispose()`
  calls `renderer.clear()` — the same lifetime deal `PaneLegendBinding`
  has.
* **A realize-owned renderer goes stale exactly as A5.2 measured.** The
  interactor holds `self._gizmos` for the life of its observers; rebuilding
  the renderer per realize either strands that reference or forces a
  re-install per realize for no gain.
* **`refresh()` is already a reconciler** with an in-place `update_layer`
  fast path over constant topology (4 quad points, 11 arrow points). That
  fast path only pays off with a long-lived owner; rebuilt per realize it
  is a fresh `add_layer` every time — the 0081 L2 flicker the module was
  written to avoid.
* **The bbox is realize's to supply.** `gizmo_geometry(plane, bbox)` sizes
  the quad from a reference bbox. Realize gains
  `RealizedPane.reference_bounds` — the bounds of the **scoped cell set's
  reference points** (`scene.reference_points`, not the posed grid), which
  is additive and computed where `scoped` and `scene` are both in hand.
  Reference, not posed, on purpose: planes are world-space and the deformed
  shape moves *through* them (0083), so the gizmo must not breathe while
  the scrubber plays. Accepted consequence: at a large deform scale the
  quad may not span the deformed cross-section. That is the plane's
  semantics, not a sizing bug.

**Who calls `refresh()`, and on what signal:**

| signal | call |
|---|---|
| every reconciler flush (`MeshPane._on_reconciled`) | `binding.bind(view, realized.reference_bounds)` → rebuilds the renderer **only** if the view identity or the bounds changed, then `refresh()` |
| a clip-only tick (the Q4 fast path, in `_flush`) | `reclip_pane` → `refresh()` → the flush's single `render()` |
| pane dispose | `clear()` + `interactor.uninstall()` |

Note the difference from legends and why it is better: legend placement is
*outside* the signature, so `PaneLegendBinding` had to become its own
dispatcher to get a repaint. Clips stay *inside* the signature (in their own
term), so the reconciler remains the single writer to the backend and the
binding needs no dispatcher seam at all.

**Interactor seating order is load-bearing.** Legend and gizmo observers
both sit at priority 12, and a legend overlapping a gizmo wins the press
only because the gizmo's observers were added *after*
(`_clip_gizmo_interactor.py:6-9`, `:338-341`). `PaneLegendBinding` re-installs
on every realize that swaps the controller — which appends the legend's
observers *after* the long-lived gizmo's and silently inverts the
precedence. So: `PaneLegendBinding.bind` returns `True` when it installed,
and `MeshPane._on_reconciled` calls `self._clips.reseat()` on `True` —
uninstall + install of the four observers, renderer and layers untouched.

### Q3 — dissolved

The working brief asked what happens when "N panes share one view, so two
gizmo sets write one `ViewClip`". **There is no such case.** A `MeshView`
*is* a pane, 1:1 by construction: `add_view()` builds
`MeshView(pane_id=f"mesh-{n}")` and appends it to `self._panes`
(`_session.py:84-93`); `MeshView.id` returns that `pane_id`; and
`_pane_signature` reads `pane.clips` off the same object. There is no
view-sharing mechanism anywhere in the session. Gizmo ownership per-pane and
per-view are the same statement. Recorded so nobody re-raises it.

### Q4 — the drag-frame answer

**Chosen: (a). `clips` leaves the structure term and gets its own fast
path.**

    signature = (structure, clips, cursor)      # was (structure, cursor)
    structure = everything the picture is BUILT from, minus clips
    clips     = pane.clips
    cursor    = session.effective_instant(pane)

In `_flush`, after the equality guard, with `forced` and "no previous
realize" refusing as they do today:

* `structure` equal, `clips` differ → `reclip_pane(view, ledger)`:
  `_apply_clips` (the same writer realize uses — one implementation, not
  two) then `binding.refresh_gizmos()`. Falls through to the existing single
  `render()`.
* `structure` equal, `cursor` moved in-stage → A4's `restep_pane`,
  unchanged.
* Both moved → reclip first (idempotent, ~1 ms), then restep. Either
  refusing or raising falls through to the full realize **in the same
  flush**, per A4.4.
* Anything else → today's path.

The clips term covers every clip edit, not just the drag: `offset`,
`normal`, `active` (specs change), `gizmo_visible` (the eye — `refresh()`
adds or drops a gizmo's two layers), `name` (a no-op the fast path absorbs).
None of them needs a realize, because none of them changes which actors
exist — and `set_clip_planes` re-stamps every live handle, so the re-cut is
total by construction.

Why not the others:

| Rejected | Why |
|---|---|
| **(b)** drag-transient, commit on `on_drag_end` | Looks cheap, is not. The cut reaches the backend only through `_apply_clips`, which runs only inside a realize — so a deferred commit leaves the model **uncut** for the whole drag while the arrow slides: you would be dragging a gizmo, not a section plane. Making the picture follow means a live-cut path, which is most of (a). It also puts the in-flight pose outside the IR, so a snapshot or an inspector read mid-drag disagrees with the screen — A5.2's "third position" class. And it does nothing for rotate |
| **(c)** accept the full realize | Measured: 240.8 ms per frame, **about 4 fps**, on a one-pane bench. Not "a bit laggy" — the plane arrives roughly a quarter-second behind the cursor, and every frame tears down and rebuilds every layer and every scalar bar |
| Throttle drag events to what a realize can do | A4.6 already rejected this shape for the scrubber: it hides the defect and makes the gesture lie about the cursor |
| Move clip state back to a viewer-level controller | Reinstates the ownership §3 deliberately moved viewer → view, and re-opens snapshot/restore for planes |
| Ask the interactor to call `set_clip_planes` directly | Two writers to the backend's clip set, one of them behind the reconciler's back — the stale-picture class ADR 0084 exists to prevent |

**The measured gate**, A4 style: re-run `profile_session.py --gesture
clipdrag --panes 1 --repeat 20 --no-profile` and record the trajectory in
A7's consequences. Target `p95 ≤ 33 ms`, expected to land near the orbit
floor (19.8 ms) because the frame becomes re-stamp + `refresh()` + render.
The CI-checkable half is criterion 5 below: `realize_pane` is called **zero**
times across a 20-frame synthetic drag.

### Q5 — the scope gizmo's disposition

**`retired`.** It is not "the same shape with a different controller":

* `ScopeController` owns a **spatial axis-aligned `BBox` per
  `geometry_id`** and hides cells by writing `vtkGhostType` through
  `ElementVisibility` (`LAYER_SCOPE`).
* The session's `Scope(axis, names)` (`_views.py:127`) is a
  **composition-axis + names** selector — physical groups, materials,
  element types. Same word, unrelated concept.
* The session IR carries **no spatial box state at all** (no `BBox`
  anywhere under `results/session/`), and no **active geometry** either
  (`_scope_gizmo.py:64-66` draws for the active geometry only; a session
  pane has one FEM scene).
* Its interactor wants `box_for(geometry_id)` / `set_scope(geometry_id,
  box)` / `show_gizmo` — three members the adapter pattern cannot serve
  off existing state.

Porting it means new §3 IR (a spatial region on `MeshView`), a snapshot
field, a resolve-time filter, and an inspector context — a §3/§4 widening
argued at the ADR bar, not a slice of R1. A2.2 is the precedent for
recording a disposition instead of deleting working code: the old viewer
keeps its scope gizmo.

Registry entry (`tests/viewers/test_session_capability_parity.py`):

```python
"install_scope_gizmo_interactor": {
    "status": "retired",
    "why": "ADR 0098 A7 Q5 — the scope GIZMO drives a spatial BBox per "
           "geometry (core/scope_controller.py, LAYER_SCOPE ghosts); the "
           "session's Scope (§3) is a composition axis plus names. Same "
           "word, unrelated concept: the session IR carries no spatial "
           "box and no active-geometry notion, so there is nothing to "
           "adapt. A spatial region on MeshView would be a §3/§4 "
           "widening and needs its own amendment. The old viewer keeps "
           "the gesture.",
},
```

---

## 4. The design

### 4.1 Modules

| file | role |
|---|---|
| `viewers/session/_clip_control.py::ViewClipController` | **landed** — the five-member contract over a `MeshView` |
| `viewers/session/_clip_bind.py::PaneClipBinding` | **new** — owns controller + renderer + interactor for one pane |
| `viewers/session/_realize.py::reclip_pane`, `RealizedPane.reference_bounds` | **new** — the shared clip writer and the gizmo's bbox |
| `viewers/session/_reconciler.py::_pane_signature`, `_flush` | **changed** — three-term signature, reclip branch |
| `viewers/session/_pane.py::MeshPane` | **changed** — construct, bind, reseat, dispose |
| `viewers/session/_inspector.py::MeshInspectorPage` | **changed** — the Section planes section (§4.4) |

### 4.2 Viewport — what the gizmo looks like (unchanged from 0083 S2)

```
 ┌──────────────────────── mesh-1 ─────────────────────────┐
 │                                                          │
 │        ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                   │
 │        │  translucent quad (blue, 25%) │                 │
 │        │        ●━━━━━━━━━━━━━▶        │  ← arrow points │
 │        │      centre        cone tip   │    at the half  │
 │        └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘    that SURVIVES │
 │                                                          │
 │  quad / shaft → translate (SIZEALL)                      │
 │  cone tip     → rotate    (HAND)                         │
 │  miss         → falls through to pick / camera, no abort │
 └──────────────────────────────────────────────────────────┘
```

No new pixels are designed here. `clip_exempt=True` and `pickable=False`
carry over unchanged, and the legend still wins a literal overlap.

### 4.3 States

| pane state | gizmo layers | interactor | legend precedence |
|---|---|---|---|
| no clips | none (`show_gizmos` derives `False`) | installed, hit-tests an empty set, aborts nothing | n/a |
| clip(s), all `gizmo_visible=False` | none | as above | n/a |
| clip(s) visible | 2 layers per visible plane | live | legend wins overlap, re-seated after every legend re-install |
| offscreen / headless backend | layers still emitted (harmless) | `install_*` returns `None`; pane simply has no gesture | n/a |
| pane disposed | `clear()` | `uninstall()` (withdraws its cursor memo) | n/a |

### 4.4 Reach — the Section planes section of the inspector

Per 0088 D2 (the inspector is the outline's property editor) and 0098 §9
(empty things are **Add** actions). The outline stays pane-granular.

Empty (0087 INV-2 — no offset field for a plane that does not exist):

```
 Inspector ── mesh-1 ─────────────────────
   Deform            ▸
   Slots             ▸
   Section planes
     + Add plane                        ⌄   (⌄ = X | Y | Z | view normal)
```

Occupied:

```
 Inspector ── mesh-1 ─────────────────────
   Section planes
     ◈ Plane 1        [x] cut  👁 on   ⇄  🗑
         offset  ├──────●──────┤  2.400 m
         normal   X   Y  [Z]  view
     ◈ Plane 2        [ ] cut  👁 off  ⇄  🗑
     + Add plane                        ⌄
```

* `[x] cut` = `ViewClip.active`; `👁` = `gizmo_visible`; `⇄` = `flipped`;
  `🗑` = `remove_clip`.
* The offset slider and the selected plane's controls render only for
  planes that exist (INV-2). No master "show gizmos" checkbox (Q1).
* One section, sectioned form, no inner title beyond the section label
  (0087 INV-1/INV-3).
* **Density discipline during a drag**: the section reconciles by
  `plane_id`, it does not rebuild. A drag frame writes one spin-box value
  and one slider position. Rebuilding rows at 30 Hz would flicker the
  focus mid-gesture — the same hazard `_outline.py:205-213` documents for
  pane rows, which is why the plane list is *not* outline children.

---

## 5. Interaction walkthroughs (0088 D7 style)

**W1 — cut the model and slide the plane.** Boot: outline, inspector,
scrubber (0088 D3), one grey mesh pane, no clips, no gizmo.
Inspector → Section planes → `+ Add plane ⌄` → **Z**. `view.add_clip` ticks
→ structure unchanged, **clips** differ → *fast path*: `set_clip_planes`,
`refresh()` adds two layers, one render. The model is cut and the quad+arrow
appear. Press on the quad → priority 12 claims the press (pick and camera
never see it). Drag → per move `set_offset` → `set_clip` → tick → fast path.
The cut and the quad follow the cursor. Release → `on_drag_end`; nothing
else fires, because nothing changed. Dock states unchanged throughout.

**W2 — hide one gizmo, keep the cut.** Click `👁` on Plane 1 →
`set_clip(gizmo_visible=False)` → clips term differs → fast path →
`refresh()` removes that plane's two layers and its hit-test entry. The cut
stays. With Plane 1 the only plane, `show_gizmos` derives `False` and the
renderer draws nothing — identical to the old viewer's master switch off.

**W3 — two panes, two independent cuts.** Outline → `+ New view` → mesh-2.
Each pane owns its `MeshPane`, `PaneClipBinding`, controller and renderer;
each reconciler's clips term reads its own `pane.clips`. Dragging mesh-1's
plane leaves mesh-2 untouched — and costs mesh-2 nothing, because its
signature did not move (criterion 12).

**W4 — a legend over a gizmo.** Fill the contour slot (full realize; the
legend binding installs a new interactor, the pane reseats the gizmo's
observers). Drag the bar where it overlaps the quad: the legend takes the
press. Drag the quad 20 px away: the plane moves. Without the reseat, the
second gesture wins both.

**W5 — scrub with a cut and a pose.** Deform on, contour filled, one active
plane. Scrub: cursor term only → `restep_pane`, 17.7 ms, gizmo untouched
(reference bounds, so the quad does not breathe while the structure sways).
The deformed shape moves through the fixed plane — 0083's stated behaviour.

**W6 — save and reopen.** `session.snapshot()` carries the clips including
`gizmo_visible` (`_snapshot.py:231-241`). Restore → `_adopt_clips` keeps the
`plane_id`s the gizmos address planes by → first flush is a full realize →
`bind` sees new bounds, builds the renderer, `refresh()` draws the same
gizmos in the same places. No new snapshot field anywhere.

---

## 6. Acceptance criteria

Numbered, each with the mutation that must break it.

1. **The gizmo exists on the session path.** A pane whose view has one
   visible clip emits exactly two layers with the `clip_gizmo:` prefix, and
   both are `clip_exempt` and non-pickable.
   *Mutation: drop the `refresh()` in `PaneClipBinding.bind` → no layers.*
2. **The interactor is installed and hit-tests the drawn geometry.** A
   synthesized press at the projected quad centre starts a translate; a
   press 3 px outside every gizmo starts nothing and does **not** abort.
   *Mutation: remove `install_clip_gizmo_interactor` → criterion 2 fails
   and the parity registry's `tested_by` points at a test that no longer
   passes.*
3. **A drag writes the view, not a shadow copy.** Press-move-release changes
   `view.clips[0].offset`, and `session.snapshot()` round-trips the new
   value.
   *Mutation: make `ViewClipController._set` swallow everything instead of
   only `KeyError` → a bad write disappears and the offset does not move.*
4. **The gizmo survives a full realize.** Fill a slot (forcing a realize)
   mid-session: the renderer object is the same instance, its handles are
   the same handles, and the layers are still on the backend.
   *Mutation: add the renderer's layers through the `LedgerBackend` instead
   of the raw backend → `_teardown` sweeps them and the gizmo vanishes on
   the next slot edit.*
5. **A clip-only tick costs zero realizes.** Over 20 synthetic drag frames,
   `realize_pane` is called 0 times and `render()` exactly 20 times.
   *Mutation: put `pane.clips` back in the structure term → 20 realizes.*
6. **The fast path is bit-parity with a realize.** After a drag, capture the
   backend's clip specs and every gizmo geometry, then `schedule_forced()`
   and capture again: identical.
   *Mutation: stub `reclip_pane` to a no-op → the forced realize produces a
   different cut, and the body fails. (The oracle is itself mutation-tested,
   per A4.3(1).)*
7. **Refusal is safe.** A `forced` flush, a first flush, and a backend
   without `set_clip_planes` all take the full path without raising; a
   `reclip_pane` that raises reports `pump.session_reclip` and falls through
   to a full realize **in the same flush**, never leaving a half-cut pane.
   *Mutation: remove the fall-through → a raised reclip leaves the previous
   flush's cut with the new record.*
8. **Legend precedence survives a realize.** Criterion from W4, asserted on
   observer order or on the outcome of an overlapping synthesized press
   after a slot edit.
   *Mutation: remove `reseat()` → the gizmo takes a press over the bar.*
9. **Two panes are independent.** Dragging pane 1's plane does not change
   pane 2's clips, its layers, or its signature.
   *Mutation: key the binding off the session rather than the view → pane 2
   re-cuts.*
10. **`show_gizmos` derives exactly.** For all 2^n flag combinations over
    n ≤ 3 planes, the drawn set equals `{P : P.gizmo_visible}`.
    *Mutation: return a constant `True` → the all-hidden case draws.*
11. **Dispose is clean.** `MeshPane.dispose()` removes every gizmo layer,
    uninstalls the observers and withdraws the cursor memo; the pyvista
    open-plotter count returns to baseline (the
    `test_pane_dispose_releases_context.py` shape).
    *Mutation: drop `renderer.clear()` → layers outlive the pane.*
12. **Reach.** The inspector's Section planes section adds a plane with no
    Python, and the added plane's gizmo appears in the same gesture. An
    empty section renders an Add action and **no** disabled offset control
    (0087 INV-2).
    *Mutation: render the offset row with no plane → INV-2 assertion fails.*
13. **The parity registry moves in the same commit.**
    `install_clip_gizmo_interactor` = `ported`, `where =
    session/_clip_bind.py::PaneClipBinding`, `tested_by =
    tests/viewers/test_clip_binding.py`; `install_scope_gizmo_interactor` =
    `retired` with the Q5 `why`; `test_the_known_gaps_are_exactly_the_two_gizmos`
    becomes `missing == []`.
    *Mutation: leave either entry `missing` → the change-detector fails,
    which is what it is for.*
14. **The measured gate.** `clipdrag` p95 ≤ 33 ms on `c010_clip_gizmo_drag`,
    `--panes 1 --repeat 20 --no-profile`, with the before (240.8 mean /
    252.2 p95) and after recorded in A7's consequences, harness kept.

---

## 7. Open questions for the owner

1. **Is the inspector's Section planes section in R1, or is R1 gizmo-only?**
   *Recommended default: in R1.* Closing R1 with planes that only Python can
   create reproduces the exact A6.1 defect class the parity gate was built
   to end, and the workshop loop's acceptance test forbids a step that needs
   Python.
2. **Does A7 also fix the six-plane cap (§2.5)?**
   *Recommended default: no — record it in A7's "still owed" and fix it in
   the same slice as the inspector's Add action, where the refusal has a
   place to be shown.* A cap enforced with nowhere to say "refused" is a
   silent drop by another name.
3. **Pick coherence under a cut (§2.5)?**
   *Recommended default: separate follow-up, named in A7.* It is a
   `PanePick` change (`view.clips` → an `is_clipped` filter over hits), it
   has its own oracle, and bundling it would make R1's parity gate ambiguous
   about which change moved the picture.
4. **Rotate on a scoped pane.** The quad sizes to the scoped cell set's
   reference bounds, so changing scope resizes the gizmo (a full realize
   anyway). *Recommended default: accept.* The alternative — the whole
   model's bounds — draws a quad over geometry the pane does not show.
