# ADR 0098 — Results presentation is a `ResultsSession` of views, not geometries of diagrams

**Status:** Accepted (2026-08-16) — design ratified in the owner
workshop the same day; record revised the same day after adversarial
review. Implementation is under way: S0, S1-A, S1-B and S2 have
merged. Qt layout of the pane host — *Open question 1* — was
**resolved in Amendment 1** (2026-08-17), the gate the implementation
plan puts before S3, and **re-decided in Amendment 3** (2026-08-19):
the central-widget host does not survive its own first show, so the
panes are dock panels.

Supersedes the **product ontology** of [ADR 0058](0058-concurrent-geometries.md)
(a `Geometry` as the Results scene instance) and its
one-global-time-cursor resolution (its open question 5): per-pane
time is the linked/unlinked model of §7.
Amends [ADR 0088](0088-viewer-window-architecture.md) D1–D3 and
D6–D7 where they concern the viewport and the Plots dock: the
centre is N panes, not one viewport; plots are panes, so
`dock_results_right` joins the retired objectNames (one schema
bump, per D6) and D2's inspector contexts restate onto pane rows
(§9). Chrome — outline, inspector, scrubber — otherwise stays 0088.
Amends [ADR 0081](0081-legend-is-a-view-annotation.md):
scale-as-view-annotation survives, per pane; the causation law is
new here (INV-LEGEND-1 — a colour scale is caused by a
colour-mapped slot, never by deform); 0081's
`(geometry, component)` key, the diagram register/unregister
contract, and per-diagram `show_scalar_bar` are superseded.
Amends [ADR 0083](0083-results-viewer-section-planes.md): the clip
machinery is kept; ownership moves viewer → view (§3); clip state
joins the session snapshot; persisted-cut auto-load moves to
clip-on-a-view (Consequences).
Amends [ADR 0094](0094-agent-assess-and-viewer-render.md) INV-10
for snapshot realization only: the closed picture space re-keys
from `view=` tokens to the §4 slot catalog (a new slot is an
amendment here, same review bar); `results.render(view=)` keeps
its four tokens.
Does not amend [ADR 0014](0014-viewer-is-pure-h5-consumer.md),
[ADR 0042](0042-render-backend-seam.md),
[ADR 0056](0056-viewer-state-and-event-contract.md) —
`ActiveObjects`' focus roster follows the new nouns (active pane,
pick target), ownership rules unchanged —
[ADR 0084](0084-results-viewer-cascade-freeze.md),
[ADR 0090](0090-scalar-bar-design.md) — typography, chip, and
format stand; its slot-allocation and containment rules apply per
mesh pane — or [ADR 0095](0095-apegmsh-studio.md) INV-6 (no Qt for
agents; `setup=` stays off the MCP per Amendment 1 and 0094
INV-10).
Narrows Results pick targets relative to
[ADR 0045](0045-selection-and-pick-contract.md): nodes or Gauss
only; element and fiber pick targets retire in this window.
Selection stays in 0045's one store — `ResultsSession.selection`
is the owner-facing surface of the one `SelectionState` /
`SelectionLog`, never a second store (INV-5).

apeCAD is a **principle source** for the stack split (Python document
is the product; GUI is a client). It is not a dependency. No apeCAD
type is imported. B-rep selection does not enter this window.

## Context

The Results desktop viewer organises the picture as

```
Geometry (mesh instance) → Composition → Diagram (kind registry)
```

Deformation is per-geometry state (0058), but the product still
treats a mesh copy as the noun and hangs pictures on it as siblings.
That inversion is the source of the chronic classes:

- a legend bound to the warp instead of the painted field
  (historical — retired by 0058 S4 + 0081; the ontology that
  produced it survived)
- hide an element, keep its outline (fill / edges / CAD as separate
  objects)
- hide a contour, grey mesh gone or leftover wire (two fills)
- plots that only exist from Python, or as an orphan dock
- selection that never became the author of those plots
- `results.viewer()` as both the document and the window

The owner workshop locked a replacement ontology. This ADR records
it as the presentation library. `Results` (the data broker) is
unchanged.

## Decision

### 1. Two libraries, stacked

`Results` answers “what did the solver write?”

`ResultsSession` answers “what is on screen, at which time, with
which pictures, with which pick?”

```
Results                         data broker (keep)
    └── ResultsSession          presentation library (this ADR)
            panes[]             MeshView | PlotView
            time, time_linked
            selection           nodes XOR Gauss
            realize()           scene IR
                    ├── Qt client          session.show()
                    ├── still / matplotlib session.render() / results.plot
                    └── Studio MCP         stills, pin, report — no Qt
```

VTK actors, docks, and widgets are not truth. The session is.
Tessellation is a projection.

Public entry:

```python
s = results.session()     # presentation exists; no window
s.show()                  # Qt client
s.render("a.png")         # still of a pane (by pane id)

results.viewer()          # sugar: results.session().show()
                          # default session = one empty mesh view
```

Panes carry a stable `id` (and optional name) in the S0 IR;
`render`, the MCP verb, the outline, and the snapshot address a
pane by it. With one pane, `render` needs no id.

`blocking=`, notebook subprocess, and `APEGMSH_SKIP_VIEWER` stay on
`show()` / `viewer()`. They host the window; they do not define views.
`restore_session` / `save_session` load and save a **session
snapshot** (panes, slots, time, style, clip, plots, selection), not
Geometry / Composition / Diagram JSON. On an old `schema_version`
the new restore skips with a one-line notice (no prompt) and
renames the file aside (`.legacy`) so save-on-close cannot destroy
the optional importer's input. `viewer(cuts=)` is retired; cuts
remain `apeGmsh.cuts` / clip-on-a-view.

The director, geometries, and diagram-kind registry are not a public
API. Scripts configure `s`. The window projects `s`.

### 2. Three authors, one session

Python and the Qt app write the same `ResultsSession`; Studio MCP
reads and realizes its snapshots — two writers, one realizer, one
session. The Qt app must be **sufficient alone** to add
views and graphics one at a time. If that loop needs a script, the
architecture has failed.

| Author | Verbs | Must not |
|---|---|---|
| Python | `session()`, `add_view`, slots, `add_plot`, `show` | Treat `ResultsViewer` as the document |
| Qt app | New view, fill/clear a slot, pick, new plot, style, time link | Bypass the session (actor mutations) |
| MCP / Studio | Realize a session snapshot (`render` / `animate` / `results_pin` / `emit_report`) | `viewer()`, Outline clicks, `setup=` |

A contour clicked in the window is the same `v.contour = …` a script
would set. An agent does not click the outline; it realizes the
session (still, animation, pin). `results_pin` may pin Results
**and** a session snapshot so a report shows what the human was
looking at, without opening Qt.

Agents still judge with `assess()` + stills, not `viewer()` (0094).

### 3. Mesh view — analysis mesh only

A mesh view is a pane on the **analysis mesh**. There is no BRep in
Results. CAD belongs to the model window.

```
MeshView
  id         stable pane id (+ optional name)
  scope      one axis — physical groups | materials | element
             types — with one or more checked names, or all
  deform     off | on (field, scale, mode?)  # pose, never a picture
  time       stage + step                # ignored when linked; none for a mode pose
  style      mesh, outlines, nodes, gauss    # independent toggles
  clip       section planes of this view     # 0083 machinery, view-owned
  overlay    undeformed off | on         # outlines style, scale 0
  slots      at most one occupant per category
  legends    derived from occupied colour-mapped slots; hide does
             not clear the slot
```

**INV-MESH-1 One cell set.** Scope chooses elements. Fill, interior
grid, outlines, node glyphs, Gauss glyphs, and every slot are
functions of that set on the current pose.

**INV-MESH-2 One surface.** Grey if the contour slot is empty; the
contour if it is occupied. A contour does not sit on a second grey
mesh.

**INV-MESH-3 Edges are derived.** Hide a cell → its face, grid line,
and outline all go. Mesh off + outlines on is a style of cells that
are still on, not a ghost of cells that were removed.

**INV-MESH-4 Four style buttons** (view chrome, not slots):

| Button | Draws |
|---|---|
| Mesh | Interior element grid on the surface |
| Outlines | Element / feature boundaries |
| Nodes | Node glyphs (required to click-pick nodes) |
| Gauss | Integration-point glyphs (required to click-pick Gauss) |

Gauss-as-button is **locations**. Gauss-as-slot is **values** on
those points (and a legend). They must not draw two clouds.

Boot of a view, no slots: the analysis mesh, grey, mesh + outlines
on, nodes/Gauss off, no legend. That is a valid picture.

ADR 0089’s three line weights remain **style of this mesh**, not
three independently owned objects.

Undeformed overlay, if on (`overlay` in the IR), is this mesh at
scale 0, drawn in the Outlines style, same cell set — not CAD, not
a second Geometry. Comparing two poses for real is a second view.

### 4. Result slots — closed catalog, unique per category

Pictures occupy unique categories. Different categories stack. Two
occupants of the same category in one view are illegal. Filling an
occupied slot **replaces** the occupant.

| Slot | Occupant | Colour-mapped | Inside the slot |
|---|---|---|---|
| `contour` | One heatmap | yes | Quantity + **averaged \| unaveraged** |
| `vector` | Arrows | yes | Quantity: principals, \(S_x\), strains, displacement, … |
| `gauss` | Values at IPs | yes | Quantity (own picker; shares the contour scale when the field matches) |
| `line` | Member diagrams | no — amplitude fill | One component v1: \(N\) / \(V\) / \(M\) / torsion |
| `sand` | Sand field | yes | Quantity |
| `loads` | Applied loads | no — uniform glyphs | Pattern / stage |
| `reactions` | Support reactions | no — uniform glyphs | — |

The colour-mapped column is the legend law's domain (§5): only a
*yes* slot emits a scale. "Field matches" = same quantity and
component; averaging is contour display state and does not break
the match (Gauss values are unaveraged by nature).

Deform is not a slot. Default deform field is displacement;
velocity, acceleration, and mode shape are legal fields of the pose,
not pictures. A mode shape carries a mode index in the pose
(`field, scale, mode`) and has no `(stage, step)`: a mode-posed
view is frozen under the time link — the scrubber moves instants
and a mode has none. Mode animation is a later widening.

v1 line uniqueness matches contour: one component per view. A later
widening may put \(N+V+M\) as options *inside* the one slot. That is
not a second category.

Fibers, shell layers, isochrones, spring-as-its-own-kind wait. They
do not open an eighth slot. `principal_glyph` and `vector_glyph`
collapse to `vector`. `line_force` is the `line` slot — v1, not
deferred. `section_cut` is not a slot (cuts / clip).

Inapplicable slots (line on a solids-only scope) are empty/disabled,
not a crash.

Averaged vs unaveraged is a property of the contour slot, not an
eighth category. Gauss markers remain the other slot.

### 5. Legends — caused by slots, owned by the view

**INV-LEGEND-1 Cause.** `legends = f(occupied colour-mapped slots)`
— the *yes* column of §4. Deform is not a slot. Empty and
non-colour-mapped slots emit nothing.

**INV-LEGEND-2 Presence.** Turning a colour-mapped slot on creates
its legend. Turning the slot off destroys it.

**INV-LEGEND-3 Independence.** Hide/show and place/size live on the
legend as view chrome. They do not live on the slot. Hiding the
scale does not hide the picture.

**INV-LEGEND-4 Identity.** The scale names the slot quantity. Warp
the mesh and contour stress → the legend says stress.

**INV-LEGEND-5 Per view.** Each mesh pane carries the legends of
*its* slots.

Same quantity on two slots (contour of S + Gauss of S) → one scale.
`show_scalar_bar` on a diagram style is not representable.

Oracle still: deform on, every slot empty → warped mesh, **zero**
legends.

### 6. Plot view

A plot is a pane, not a dock of a contour.

```
PlotView
  kind       history | path | xy
  series[]   live queries (source + quantity)
  cursor     an instant
```

`source` is a node, a Gauss point, a label, a physical group, or the
session selection. Python `add_plot(...)` and “select → New plot”
write the same `PlotView`. “Select → New plot” **copies** the
membership into the `PlotView` at creation — a live alias to the
session selection is not a v1 source; “live queries” means live
against `Results` at the cursor, not live membership. Several
series on one chart are one plot view.

`results.plot.history` remains the matplotlib still; it uses the
same query.

### 7. Time

An **instant** is a `(stage, step)` pair. Dragging the curve or the
scrubber snaps to the nearest recorded step. A mode-posed view (§4)
has no instant and is frozen under the link.

- **Linked** (session `time_linked=True`): one instant. Scrubber,
  every mesh view, every plot cursor. Drag the curve → meshes move.
  Drag the scrubber → the cursor rides the curve.
- **Unlinked:** each pane keeps its own instant.

A history plot shows the whole record; the cursor is time. A path
plot is evaluated at the current instant.

If the link is on and moving the plot does not move the meshes, the
link is a lie.

### 8. Selection — nodes or Gauss only

Results selection is only **nodes** or **Gauss points**. Not
elements, not faces, not BRep. An element is a membership query
(“all Gauss of this hex”), not a hit.

Four writers, one homogeneous set:

1. Click (individual)
2. Window (rubber-band; scoped to this view’s visible cells)
3. Outline: *Select all nodes* / *Select all Gauss* on a PG,
   material, element type, or element
4. Code

The set’s kind follows the last writer: a write of the other kind
**replaces** the set. The per-view radio below only aims clicks and
windows in that view; it neither owns nor clears the set, and two
views may aim at different targets over the one set. The set lives
in 0045’s `SelectionState` / `SelectionLog` — the session is the
owner-facing surface, not a second store.

That set **is** the `source=` of plots. If selection does not
produce a plot without a Python snippet, selection is still broken.

Pick target is a radio on the mesh view: Nodes | Gauss. Clicks and
windows hit that target. Query-from-outline may fill the set with
glyphs off; glyphs on is how you see it. Click-pick requires the
matching style button on.

Scope (what is drawn) and selection (what is plotted) are different
knobs.

Highlights follow the current pose. Visibility toggles never change
the cell set or the selection set.

### 9. Outline

Two jobs, one tree:

- **Panes** — mesh views and plot views. Select a pane; the
  inspector edits that pane (deform, slots as Add / change /
  clear, pane time — set it here; the link ignores it, §7).
- **Model composition** — physical groups, materials, element
  types. Check = this mesh view’s **scope**. Action = **select all
  nodes or Gauss** of that category.

v1 scope is one composition axis — physical groups, materials, *or*
element types — plus one or more checked names on that axis (or
all), not a boolean `concrete AND hex`.

Empty slots in the inspector are **Add** actions. An occupied slot
is change quantity or clear. New view / new plot live on the
outline. That is the discrete visual loop.

### 10. Clients and reuse

| Client | Job |
|---|---|
| `session.show()` | Qt: tiled panes, inspector, scrubber, link-time, style buttons |
| `session.render()` | Offscreen still of a pane (test oracle, Studio, assess) |
| `results.plot` | Matplotlib still of the same queries |
| Studio MCP | Stills and names; session snapshot; does not drive Qt |

**Keep:** `Results`; scene IR + `RenderBackend`; geometric pick
(screen → cell), then interpret as node/Gauss; FEM grid from
results (analysis mesh only).

**Replace as product nouns:** Geometry, Composition, unlimited
diagram stack, deform-as-graphic, element pick in this window, BRep
in this window, plots that only exist from Python, the Qt class as
source of truth.

Reuse VTK and the seams. Do not reuse the old ontology. The
presentation library lives in `apeGmsh.results.session`. The Qt
client lives in `apeGmsh.viewers` as a projection. No third package.

### 11. Build order

Step 2 not green means the window does not grow.

| Slice | What | Verify |
|---|---|---|
| S0 | `ResultsSession` IR + slot uniqueness + selection type + time-link laws | Tests, no Qt |
| S1 | `realize` → scene IR + stills | Layer assertions are the primary oracle (deform without legend; contour averaged/unaveraged; hide cell ⇒ edge gone; selection → plot query); pixel stills secondary, skip ≠ pass (0094) |
| S2 | Qt: one mesh pane as a projection of the session; discrete Add slot | Headless: widget event → session diff → repaint (offscreen Qt); the human loop with no Python is the acceptance |
| S3 | N panes (pane-host layout — open question 1) + per-view scope + style buttons | Two views, different slots/scopes |
| S4 | Plot views, node/Gauss pick + window + outline select-all, time link | Pick → plot; linked cursor moves meshes |
| S5 | Session snapshot; `results_pin` may pin it; MCP `render` of a snapshot | Agent still of what the human built, no Qt |
| S6 | Retire Geometry / Composition / Diagram as the public surface; flip `viewer()` | Callers use `session()` (the `show_web` hatch excluded) |

`viewer()` keeps opening today’s window through S5; it flips to
`session().show()` at S6, together with the `python -m
apeGmsh.viewers` subprocess path and the restore/save snapshot
semantics. After the flip `viewer()` returns the `ResultsSession`.
During S2–S5 the new window is `results.session().show()`.

ADR 0084’s one-reconciler discipline applies to session → realize →
paint — a **new** reconciler under 0084’s guard discipline; the
Dispatcher stays with the old ontology. Dual STEP/RENDER and silent
pumps stay forbidden.

## Alternatives rejected

| Rejected | Why |
|---|---|
| Keep Geometry → Composition → Diagram and restyle Qt | The nouns are the bug |
| Deform as a graphic / diagram kind | Pose is not a picture; it births the wrong legend |
| Unlimited stacked diagrams of the same kind | Two contours are unreadable; uniqueness is per category |
| Abaqus one-field viewport as the only model | The owner requires stacked *categories* and N views |
| BRep in the Results window | Ghost CAD; cannot follow DOFs; wrong viewer |
| Fill / edges / CAD as sibling actors | Ghost outlines |
| Plots as a right dock owned by a diagram | Orphan UI; code-only path |
| Element pick in Results | Plots consume nodes or Gauss; element is a membership query |
| Studio CAD/Qt as the results author | Violates 0095 INV-6; agents get stills of the session |
| Import apeCAD types / B-rep outliner | Wrong domain; would restore BRep |
| `viewer()` returns a god `ResultsViewer` to mutate | Window is a client; configure `session()` or the UI |
| Eighth slot for fibers / layers / isochrones in v1 | Catalog stays closed |
| MDI subwindows as the pane host | Rejected in the workshop as daily chrome; splitters are the baseline (open question 1 keeps only the detail) |

## Consequences

- `results.viewer()` remains the human one-liner and becomes
  `session().show()` with a default empty mesh view (flip at S6,
  §11).
- Session JSON schema is a new snapshot of panes/slots, not a
  compatible Geometry tree. Restore of old `.viewer-session.json`
  is not owed; the old file is renamed aside (§1) so a one-shot
  importer — optional and unnamed here — keeps its input.
- Web/trame (`show_web`) is a later client of the same session, not
  a second ontology. Until it lands, `viewer()` notebook fallback
  may keep today’s web path as a compatibility hatch — the director
  survives past S6 as an implementation detail of that hatch,
  excluded from S6’s gate. `export_animation(setup=)` stays the
  0095 INV-11 human/script escape hatch; the director handle it
  receives becomes internal at S6 with no compatibility promise.
- Tests of “diagram kinds” that encode Geometry/Composition move
  onto session IR + stills — ~80 test files import
  `viewers.diagrams` today, so the move is staged with the slices
  and the old suite stays green until S6. Scene-IR layer assertions
  are the S1+ oracle; pixel stills are secondary (skip ≠ pass).
- Shipped kinds with no slot — `fiber_section`, `layer_stack`, the
  three isochrone kinds, `spring_force` — do not regress silently:
  S6 does not complete until each has a recorded disposition (a
  slot-widening amendment, explicit retirement, or internal
  survival behind the `show_web` hatch).
- h5-persisted section cuts keep their auto-load contract
  (kwarg-wins, H14) as clip-on-a-view: a new session boots with
  persisted cuts as view clips. The `section_cut` preview picture
  retires with the kind — the cut plane is clip state (0083
  gizmos); a resultant display is a later widening, not owed here.
- Docs and the `apegmsh` skill are in-scope at S6:
  `docs/api/viewers.md` (mkdocstrings of `ResultsViewer`), the
  results design and how-to pages, and the skill’s `results.md` /
  `opensees-bridge.md` move to the session surface. `mkdocs build
  --strict` (0079) is the gate.
- MCP grows only habitat verbs over the session snapshot (`render`
  a pane, pin a session). It does not grow `setup=`, Outline
  clicks, or session-mutation verbs. For snapshot realization the
  closed picture space is the §4 slot catalog (amended 0094
  INV-10); a new slot is an amendment here, same review bar.

## Open questions

1. **Pane host (UI, not IR).** How N mesh/plot views are tiled in
   Qt. The workshop rejected MDI subwindows as daily chrome and
   recommended nested splitters as the baseline; what remains open
   is the detail — splitters vs tab+split editor area vs a mosaic +
   plot strip. Chrome (outline / inspector / scrubber) stays 0088
   as amended above. **Resolved in Amendment 1** (nested
   `QSplitter`s, auto-tiled by `T(N)`; one `QtInteractor` per mesh
   pane).
2. **Averaging region.** v1 averaged contour is global on the
   visible cell set. Averaging only within a material / PG is a
   later widening of the contour slot.
3. **Web client.** Same session, later ADR. The director survives
   only as the `show_web` hatch’s internal implementation until
   then (Consequences); it does not survive as an ontology or a
   public surface.

## Amendment 1 (2026-08-17) — the pane host

Append-only. §1–§11, INV-MESH-1…4, INV-LEGEND-1…5, the slot catalog
and the rejected-alternatives table all stay. This amendment closes
**Open question 1** and nothing else.

It is **UI, not IR.** No session record changes: `ResultsSession`,
`MeshView`, `PlotView`, `realize()` and the (unwritten) snapshot are
untouched. Layout is window chrome, exactly like dock geometry — it
lives in `QSettings`, never in the session. Chrome outside the centre
— outline, inspector, scrubber — stays ADR 0088 as amended by this
ADR; no dock is added, renamed or retired, so
`SessionResultsWindow._LAYOUT_SCHEMA_VERSION` stays 1.

Required before S3 starts (implementation plan, hard sequence rule).

---

### Evidence — the spike, on both oracles

The plan named "multi-interactor driver quirks" the single biggest S3
risk and demanded a prototype on Windows **and** under CI Mesa before
this amendment was written. It was built: **N pane-owned
`QtInteractor`s in nested `QSplitter`s (an h-splitter of v-splitters)
is viable.**

| Oracle | Result |
|---|---|
| Windows, real GPU, 4 interactors | Nine probed failure modes pass: four distinct renderers, independent scenes, independent cameras (moving pane 0's camera left panes 1–3 untouched), exactly one scalar bar per pane, per-pane screenshots, splitter-resize survival, runtime add/remove of a pane, **a live pane reparented between splitters keeping its camera pose and a non-flat frame**, clean teardown |
| Linux CI, `xvfb` + Mesa software GL (`qt-window-tests` lane) | 7 passed, 0 skipped |

The seven durable modes are committed as
`tests/viewers/test_pane_host_probe.py` (PR #1019) and stay after the
spike as S3's regression guard — the day a VTK or pyvistaqt bump makes
four live interactors share a camera or leak a scalar bar, the pane
host breaks and that file fails first.

The reparent mode was added after adversarial review: `T(N)` re-tiles
at every column-count change (N = 2, 5, 10…), which **moves** a live
pane between splitters, and the original probe never did that — it
closed one interactor and built a fresh one, a different operation.
Reparenting a GL-native widget destroys and recreates its platform
window, so this was the one lifecycle event the tiling law mandates
and no oracle had exercised. It passes on both.

Two inherited facts, recorded so they are not re-derived:

1. **Per-pane scalar bars are free.** Each plotter keeps its own
   registry (`iren.scalar_bars` keys are per-plotter — proven by
   `test_scalar_bars_stay_in_their_own_pane`). INV-LEGEND-5 needs no
   legend-routing layer, and the reconciler's existing
   `adopt_controller` per flush is already per-backend.
2. **A `QT_QPA_PLATFORM=offscreen` segfault is platform GL, not a
   pane-count problem.** Offscreen gives VTK no usable pixel format; a
   *single* interactor dies identically, on Windows and in the default
   CI lane alike. An offscreen crash during S3 is not evidence against
   the pane host. It is also precisely why `MeshPane` takes an
   injectable `backend_factory` (plan decision 7) and why half the
   acceptance criteria below run through injected backends.

**The fallback design — one render window with N VTK viewports — is
dropped.** It was the plan's stated alternative to N interactors; the
spike removes its motivation, and it costs: manual viewport-rect maths
on every resize, one renderer stack to route N cameras through, no
per-pane `screenshot()` (S1's stills law rides one screenshot per
pane), a legend-routing layer to rebuild what the per-plotter registry
gives free, and hand-written hit-region maths for S4's pick instead of
Qt delivering the event to the right widget. Nothing is bought.

*Note on framing:* the ADR's open question asks about the **layout
idiom** (splitters vs tab+split vs mosaic+strip); the plan asks about
the **rendering substrate** (N interactors vs N viewports). Both axes
are settled here.

---

### A1.1 Decision

**The pane host is one tree of nested `QSplitter`s, auto-tiled, in the
window's central widget. Each mesh pane owns one `QtInteractor`; each
plot pane owns one chart widget. No tabs, no MDI, no plot strip.**

Concretely: the central widget is a root `QSplitter(Horizontal)` of
**columns**; each column is a `QSplitter(Vertical)` of **rows**; each
row is one pane frame. Depth is exactly two. That is the geometry the
probe proved.

The host is a projection of `session.panes`, on the S2 discipline: the
session is truth, the widget tree follows. `session.add_view()` from
Python and "New mesh view" from the outline produce the same window.

**On the session path the shell builds no central interactor.** The
pane host *is* the central widget; every GL context in the window
belongs to a pane. The seam is additive and opt-in — a constructor
flag or a `set_central_widget` on `ViewerWindow` / `ResultsWindow`
that the session window passes and the old window never reaches, so
`results_viewer.py`'s construction path stays byte-identical.

This is part of the decision, not a detail deferred to S3, because
the acceptance criteria below depend on it. Today
`_results_window.py:128` builds a `ViewerWindow`, which builds a
`QtInteractor` unconditionally at `viewer_window.py:264`; S2's own
suite records the consequence — "constructing the real shell
allocates a `QtInteractor` GL context, so the full composition smoke
is qt-marked" (`tests/viewers/test_session_window.py:6-10`). Under
`QT_QPA_PLATFORM=offscreen` that context cannot be created at all
(inherited fact 2), so **seven of the `[off]` criteria below — 1, 6,
7, 8, 9, 13, 14, 15 — are only honest if the shell stops building
it.** Injecting pane backends does not help: it does not touch the
*shell's* interactor. The alternative was to re-tag those seven
`[qt]` and pay the xvfb lane for the whole host suite.

It also disposes of a risk rather than managing it: with no shell
interactor there is no orphaned central context to leak when the
host mounts (caution 1), so that caution is closed by construction.

### A1.2 The tiling law — `T(N)`

The shape is a **pure function of the pane count**, not of click
history:

```
C = ceil(sqrt(N))                 columns
cells filled row-major, in session.panes order (creation order)
column j holds panes j, j+C, j+2C, …    (top to bottom)
```

| N | Shape |
|---|---|
| 1 | one pane, no visible handle |
| 2 | two columns |
| 3 | col 1 = [p1, p3], col 2 = [p2] |
| 4 | quad — col 1 = [p1, p3], col 2 = [p2, p4] |
| 6 | 3 columns × 2 rows |

Row-major fill is chosen over column-major because it minimises
movement: **adding a pane never moves the others unless the column
count changes** (at N = 2, 5, 10, …). Closing a pane re-tiles the same
way.

Handle positions are the user's. `T(N)` decides the *shape*; splitter
ratios are dragged freely and persist (A1.5), and a re-tile disturbs
the least it can:

```
a splitter whose CHILD LIST changed  -> its ratios reset to equal
every other splitter                 -> keeps its dragged ratios
View → Reset layout                  -> everything resets
```

So adding a third pane (`T(2)` → `T(3)`, which only grows column 1)
resets that column and leaves the root's hand-dragged 70/30 alone; a
column-count change (N = 2, 5, 10…) rewrites the root's child list
and does reset it. The rule stays a pure function of the shape
change — no click history, nothing stored beyond the ratios already
in `panes/layout` (A1.5 keys `cols` and `rows` separately, which is
what makes per-splitter reset expressible).

An earlier draft of this amendment reset *every* ratio on every Add
and Close ("a tile is a tile"). Adversarial review named it the
likeliest bug report in the first week — the one-large-mesh /
one-narrow-plot arrangement this design explicitly supports is
destroyed by an Add that never touches the splitter holding it — and
it bought no determinism the narrower law does not also buy.

There is no per-Add split direction, because there is no local split:
the grid re-tiles. This is the Abaqus "tile viewports" model, not the
VS Code "split editor" model, and this window's operator knows the
former.

### A1.3 The pane frame

Every pane — mesh or plot — is a `SessionPaneFrame`: a one-row header
above the pane content.

```
┌ SessionPaneFrame ─────────────────────────────────────────┐
│ SessionPaneHeader   h = DENSITY.row_h (22 / 28)           │
│  ▦ Mesh view 1          [M][O][N][G]      (time)       ✕  │
├───────────────────────────────────────────────────────────┤
│  pane content — QtInteractor (mesh) | chart (plot)        │
└───────────────────────────────────────────────────────────┘
```

| Zone | Mesh pane | Plot pane |
|---|---|---|
| Left | kind glyph + `view.name or view.id`, section-header role (`fs_head`, 600, `text`) | same, + dim `kind` word (`history`) |
| Middle | the **four style buttons** (INV-MESH-4) as 16 px icons in 22 px hit areas | *(nothing — they do not act on a plot; 0087 INV-2)* |
| Right, reserved | per-pane instant badge when the time link is off (**S4-3**; renders nothing in S3) | plot cursor badge, same slice |
| Far right | `close` glyph | `close` glyph |

Rules:

- The pane is not a dock, so it owns its single title — 0087 INV-1's
  stated exception for surfaces that render outside a dock. It is the
  **only** title: no inner header inside the pane content.
- **Style buttons live here and nowhere else.** They are view chrome,
  not slots (INV-MESH-4); routing them through the inspector would
  cost a pane selection per toggle (0088 D2's accepted
  one-context-at-a-time cost) exactly during the gesture — comparing
  two panes — that N panes exist for. The mesh-view inspector page
  does **not** restate them: one state, one control.
- **Icons, per 0087 INV-4.** `mesh` already exists in
  `_icon_factory.py`; `outlines`, `nodes`, `gauss` are three new
  glyphs and four new Appendix A rows. Metaphors, kept distinct at
  16 px: `outlines` = closed boundary polygon, no interior lines
  (vs `mesh`, interior grid + two diagonals); `nodes` = cell corners
  as three filled dots; `gauss` = cell with a 2×2 interior dot
  pattern (vs `probe_slice`, a 3×3 line grid with no dots). Word
  labels were considered and refused: four words plus title plus
  close do not fit the 240 px width floor.
- Toggling a style button writes `session.pane(id).style` — the same
  `MeshStyle` record a script would assign. It never touches another
  pane.

### A1.4 Floors, the Add gate, the empty host

Two new `LAYOUT` fields, on that file's documented promotion path:

| Field | Value | Why that number |
|---|---|---|
| `pane_min_width` | **240** | 0090's chip is content-derived, not a fixed rect: `box_px` returns 133.5 px wide for a default vertical entry (`displacement_z`, 5 labels, `font_scale` 1.0), and grows with the component name — a 16-character name reaches ~151. Plus `MARGIN_PX` 16 that is ~150–167; 240 leaves the model visible beside its own legend even for a long component name |
| `pane_min_height` | **200** | same entry measures 154.5 px tall; height tracks `n_labels`, not the name — 5 labels 154.5, 6 labels 180.4, 7 labels 206.3. With `MARGIN_PX` the floor contains the default 5-label legend (170.5) and a 6-label one (196.4) but **not** a 7-label one (222.3), which needs a `font_scale` change. 200 is the "default legend fits" floor, not "any legend fits" |

`splitter_handle_width` (4) is reused, not re-invented. Every splitter
sets `childrenCollapsible(False)`; a pane can be dragged small, never
to zero.

**These floors are measured against today's `_legend.py` content
boxes, which pre-date 0090 D3's chip.** D3 adds an 8 px chip pad per
side and moves containment onto the chip rect — +16 px per axis —
under which the 200 floor no longer contains even a 6-label legend
(196.4 + 16). When 0090's implementing phase lands, these two numbers
are revisited; they are not a permanent contract.

**Add gate.** "New mesh view" / "New plot" are *enabled* only when
`T(N+1)` clears both floors at the host's current size:

```
C' = ceil(sqrt(N+1))            R' = ceil((N+1) / C')
host_w >= C'*240 + (C'-1)*4     host_h >= R'*200 + (R'-1)*4
```

Otherwise they render disabled with the reason in the tooltip — the
shipped `_SlotRow(can_add=False)` idiom, restated (0087 INV-2: a
control that cannot act does not pretend to). Practical ceilings:
1280 × 800 → 4 panes; 1600 × 1000 (the shipped default window) →
9 panes.

**The gate constrains creation only.** A session that arrives with
more panes than fit — `s.add_view()` in a script, an S5 snapshot
restore — is tiled anyway; the IR is truth and the host never refuses
a pane that exists. Qt then raises the window's minimum size. Beyond
the screen the panes clip; accepted, and named here so it is not read
as a defect.

**Zero panes is a valid session** (`ResultsSession` docstring), so the
host renders it: one `SessionPaneHostEmpty` card, one hint line, one
action ("New mesh view") — the 0088 D3 empty-state-HUD idiom, INV-2
shape, zero enabled data controls. The inspector shows its
nothing-selected hint beside it.

### A1.5 Active pane, closing, persistence

**Active pane.** One at all times (when N ≥ 1). It is the same state
as the outline's selection and the inspector's context — one state,
three renderings:

```
outline row selected  ==  host.active_pane_id  ==  inspector context
```

It lives in the viewer's `ActiveObjects` (`activeViewChanged`, ADR
0056 — 0098 already says the focus roster follows the new nouns), not
in the session IR. Rendering: **1 px `accent` border on the active
pane frame** — 0087 D1.4's "accent reserved for focus/active" — plus
a 2 px `accent` left spine on the outline row. That spine is **owed
work, not existing behaviour**: today the pattern exists only as
`QFrame#PlotPaneTabRow[active="true"]` (`ui/theme.py:1055-1060`), and
`session_outline_tree` has no QSS block at all. S3 reuses the
`PlotPaneTabRow` idiom on the outline row; it does not inherit it.
Clicking anywhere in a pane makes it active; so does selecting its
outline row.

**Closing.** The header's `close` glyph calls
`session.remove_pane(id)`; the outline row's context menu offers the
same. The widget disappears because the session lost the pane, never
before. Closing the **last** pane is allowed — it lands on the empty
card. No undo, consistent with the rest of this window.

**Persistence.** Shape is derived, so only ratios are stored, in the
session window's own scope (plan decision 8), as one JSON string:

| Key | Scope | Value |
|---|---|---|
| `panes/schema_version` | `QSettings('apeGmsh', 'ResultsSession')` | `1` |
| `panes/layout` | same | `{"n":4,"cols":[0.5,0.5],"rows":[[0.5,0.5],[0.5,0.5]]}` |

Ratios, not pixels, so a restore at another window size divides
proportionally. Applied only when `n` equals the session's pane count
and the schema version matches; otherwise equal ratios, silently (this
is chrome — `_restore_layout`'s existing silent-mismatch behaviour,
not §1's session-schema notice). Nothing is written under
`('apeGmsh', 'ResultsViewer')`. **View → Reset layout** restores the
0088 D1 dock set *and* re-tiles to `T(N)` with equal ratios.

Within a live session, ratios survive per the A1.2 law: only the
splitter whose child list changed is reset. Note that before S5 the
stored ratios are nearly inert — a fresh session boots at N = 1, so
the `n` guard rejects most restores. Their real consumers are the S5
snapshot restore and same-shape scripted sessions; this is expected,
not a persistence defect.

### A1.6 Decision table

| Question | Decision | Serves |
|---|---|---|
| Layout idiom | Nested `QSplitter`s: root H of columns, each column a V of rows, depth 2 | Open question 1; probe geometry |
| Rendering substrate | One `QtInteractor` per mesh pane; N-viewports fallback dropped | Spike, both oracles |
| Split direction on Add | None — the grid re-tiles by `T(N+1)` | A1.2 |
| Where a new pane lands | Next cell of `T(N+1)`, row-major in `session.panes` order; becomes active + outline-selected | §9, workshop loop |
| Tabbed panes | **No.** A hidden pane is a pane not doing its job; tabs are a second navigation spine beside the outline (0088 D2's own rationale) | 0088 D2 |
| Drag-reorder panes | Not in v1 — a second author of pane order beside creation order, not an IR problem (see the rejected-alternatives note) | one spine |
| Minimum pane size | `LAYOUT.pane_min_width` 240 × `pane_min_height` 200; `childrenCollapsible(False)` | 0090 D3/D4 footprint |
| Add gate | Disabled with a reason when `T(N+1)` breaches a floor | 0087 INV-2 |
| More panes than fit (Python / restore) | Tiled anyway; window minimum grows; never refused | IR is truth |
| Pane header shows | kind glyph + name · four style buttons (mesh only) · reserved time badge · close | INV-MESH-4, 0087 INV-1 |
| Style buttons' home | Pane header only; inspector does not restate them | INV-MESH-4, INV-2 |
| Active pane indicated by | 1 px `accent` frame border + outline row spine; `ActiveObjects.activeViewChanged` | 0087 D1.4, ADR 0056 |
| Pane closed by | Header `close` → `session.remove_pane`; outline context menu | §9 |
| Last pane closed | Allowed → empty-state card with one action | zero panes is valid IR |
| Layout persisted | Yes — ratios only, `panes/layout` + `panes/schema_version` in the `'ResultsSession'` scope | plan decision 8, 0088 D6 |
| Dock schema | Unchanged; `_LAYOUT_SCHEMA_VERSION` stays 1 (no dock added/renamed) | 0088 D6 |
| Legends | Per pane, per plotter; 0090 slot allocation and containment measured against the **pane's** viewport | INV-LEGEND-5 |
| Toolbar camera / screenshot / fit | Act on the **active pane** | one active object |
| Plot panes | Same host, same frame, same tiling. No strip, no dock | §6 |
| New view / new plot live | Outline only (§9). No View-menu duplicates | one spine |

---

### Wireframes

Double rule = the 1 px `accent` border of the **active** pane.
`[M][O][N][G]` = the four style buttons (Mesh / Outlines / Nodes /
Gauss).

**One pane — boot, 1280 × 800.** The host inherits the whole central
region, so 0088 D1's invariant is unchanged: 1280 − 260 − 380 = 640 =
50 %.

```
┌──────────────────────────────────────────────────────────────────────┐
│ File   View   Help                                                   │
├───┬───────────┬──────────────────────────────────┬───────────────────┤
│ T │ Outline   │╔════════════════════════════════╗│ Inspector         │
│ o │ Views     │║▦ Mesh view 1  [M][O][N][G]   ✕ ║│  Deform      [ ]  │
│ o │  Mesh v 1 │╟────────────────────────────────╢│  Contour   [Add]  │
│ l │  + New …  │║                                ║│  Vector    [Add]  │
│ b │           │║        grey mesh               ║│  Gauss     [Add]  │
│ a │ Model     │║        (valid picture, §3)     ║│  …                │
│ r │  □ Deck   │╚════════════════════════════════╝│                   │
├───┴───────────┴──────────────────────────────────┴───────────────────┤
│ Time scrubber                                                        │
└──────────────────────────────────────────────────────────────────────┘
     260 px              640 px — the pane host              380 px
```

**Two panes — after "New mesh view".** `T(2)`: two columns, 318 px
each. The new pane is column 2, active and outline-selected; the
inspector now edits it. Pane 1's legend stays pane 1's
(INV-LEGEND-5).

```
┌───────────────────────────────┐╔═══════════════════════════════╗
│▦ Mesh view 1 [M][O][N][G]   ✕ │║▦ Mesh view 2 [M][O][N][G]   ✕ ║
├───────────────────────────────┤╟───────────────────────────────╢
│  contour ▓▓▓▓▓▓       ┌─────┐ │║                               ║
│                       │ S11 │ │║        grey mesh              ║
│                       └─────┘ │║                               ║
└───────────────────────────────┘╚═══════════════════════════════╝
              ↑ 4 px handle — the only chrome between panes
```

**Four panes — `T(4)`, the quad.** Splitter tree
`H[ V[p1,p3], V[p2,p4] ]` — the probe's geometry, 318 × 333 each.

```
┌──────────────────────────────┐┌──────────────────────────────┐
│▦ Mesh view 1 [M][O][N][G]  ✕ ││▦ Mesh view 2 [M][O][N][G]  ✕ │
├──────────────────────────────┤├──────────────────────────────┤
│ contour S11  → 1 legend      ││ deform on, every slot empty  │
│                       ┌────┐ ││ → ZERO legends (§5 oracle)   │
│                       │S11 │ ││                              │
└──────────────────────────────┘└──────────────────────────────┘
╔══════════════════════════════╗┌──────────────────────────────┐
║▦ Mesh view 3 [M][O][N][G]  ✕ ║│▦ Mesh view 4 [M][O][N][G]  ✕ │
╟──────────────────────────────╢├──────────────────────────────┤
║ gauss slot   → 1 legend      ║│ scope: Deck only             │
║                       ┌────┐ ║│ (one cell set, INV-MESH-1)   │
║                       │S11 │ ║│                              │
╚══════════════════════════════╝└──────────────────────────────┘
```

**Mesh panes + a plot pane — `T(3)`.** A plot is a pane, not a strip
and not a dock (§6). Creation order Mesh 1, Mesh 2, Plot 1 puts the
plot at column 1 row 2; mesh 2 keeps the full-height right column.

```
┌──────────────────────────────┐╔══════════════════════════════╗
│▦ Mesh view 1 [M][O][N][G]  ✕ │║▦ Mesh view 2 [M][O][N][G]  ✕ ║
├──────────────────────────────┤╟──────────────────────────────╢
│ contour + deform             │║                              ║
│                       ┌────┐ │║   same model, unaveraged     ║
│                       │S11 │ │║                              ║
└──────────────────────────────┘║                              ║
┌──────────────────────────────┐║                              ║
│▤ Plot 1  history           ✕ │║                              ║
├──────────────────────────────┤║                              ║
│  tip Uy    ╱────┆ cursor     │║                              ║
│      ─────╯     ┆            │║   drag the cursor → the      ║
│                 ┆            │║   meshes move (link on, §7)  ║
└──────────────────────────────┘╚══════════════════════════════╝
```

With one mesh view the same loop gives the simpler `T(2)`: mesh left,
plot right.

**Add to a full layout.** At 1280 × 800 with four panes, `T(5)` needs
3 × 240 + 8 = 728 px of host width and only 640 exist:

```
 Outline                          status bar
┌──────────────────┐             ┌───────────────────────────────────┐
│ Views            │             │ Panes 4 · adding a fifth needs a  │
│  Mesh view 1     │             │ wider window                      │
│  …               │             └───────────────────────────────────┘
│  + New mesh view │ ← disabled, tooltip:
│  + New plot      │   "A fifth pane needs 728 px of viewport width
└──────────────────┘    (240 px minimum per pane); the window has 640."
```

No dialog, no silent failure, no shrinking below the floor.

**Last pane closed.**

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│              No views in this session.                        │
│                                                               │
│                   [ New mesh view ]                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
  one hint, one action, zero enabled data controls (0087 INV-2)
```

---

### Alternatives rejected

| Rejected | Why |
|---|---|
| **MDI subwindows** | Already rejected in the workshop as daily chrome; carried forward, not re-derived. Overlapping, z-ordered, hand-arranged windows make the operator do layout work before doing engineering work, and nothing tiles by itself. The ADR's table row stands |
| **Tab + split editor area (VS Code)** | A tabbed pane is invisible while its neighbour is up — N panes exist for *simultaneous* comparison (§3, the workshop loop). Tabs are also a second navigation system beside the outline, which is the exact defect 0088 D2 removed from the tab spine |
| **Mosaic of meshes + a plot strip** | Privileges mesh panes over plot panes and re-creates `dock_results_right` under a new name — the dock this ADR retires (§6: "a plot is a pane, not a dock of a contour") |
| **One render window with N VTK viewports** | The plan's substrate alternative. The spike removes its motivation; it costs viewport-rect maths, a legend-routing layer, per-pane screenshots and S4 pick routing, and buys nothing |
| **Local "split this pane" instead of auto-tiling** | Two shapes to reason about (the live tree and the canonical one), a persistence format for arbitrary trees, and depth > 2 with slat-shaped panes. Deferred as a widening, not needed for the loop |
| **Free-drag pane reordering** | Not a state problem — `session.panes` is already the ordered list `T(N)` consumes, and open sub-question 4 concedes the widening needs no IR change. It is refused as v1 *scope*: a drag gesture that rewrites `session.panes` is a second way to author pane order beside creation order, and the loop is not yet proven enough to need one |
| **Maximize / zoom one pane** | Refused as a second *layout* mode, not as a hidden-siblings mechanism — ADR 0088 D6's Ctrl+H focus mode already ships hide-with-restore for the docks, and this would be the same gesture applied to the tiling. What it costs is a live tree that disagrees with `T(N)`, which is the shape everything else here reads. The operator can close a pane and re-add one; revisit if the quad view proves too tight |
| **Pane layout in the session snapshot** | Layout is chrome. Putting it in the IR would make a snapshot restore reproduce someone else's monitor, and would reopen S0's records for a UI question |

---

### Cautions S3 inherits

1. **The S2 window adopts the shell's plotter and must stop.**
   `viewers/session/_window.py` injects a factory returning
   `PyVistaQtBackend(self._shell.plotter)`; with N panes each pane
   uses `MeshPane.default_backend_factory` and owns its interactor.
   The shell's own `QtInteractor` (built in `viewer_window.py:264` and
   set as the central widget at :265) then has no viewer. Do not leave
   it alive as an orphan GL context. The seam should be **additive and
   opt-in** — the old window's construction path must stay
   byte-identical (standing risk: nothing changes for
   `results_viewer.py` before S6a).
2. **pyvista's global theme is mutated *after* the shell's interactor
   is constructed** (`viewer_window.py:264` vs :271–272 — the ADR 0090
   E1 ordering defect). Pane interactors built later inherit
   `dark` / white font. Legends are safe (0090 D1 routes their colours
   through specs) and realize bakes substrate colours from the palette,
   but any other pyvista-defaulted actor in a pane is not. Fix the
   ordering, or construct panes with an explicit theme.
3. **`MeshPane.set_pane` loses its caller.** In S2 the outline click
   re-aimed the one viewport; from S3 each pane is constructed with its
   own `pane_id` and never re-aimed — the click selects. Retire or keep
   for tests; no behaviour may ride it.
4. **N reconcilers, one tick.** Each pane owns a `SessionReconciler`
   with its own coalescing timer, so a single session tick schedules N
   flushes. At 18.6 M cells that is the plan's named responsiveness
   concern multiplied. Criterion 12 pins the observable; the mechanism
   is S3's (a pane-level signature over records + effective instant +
   style + scope + clips + overlay, with a forced path for the theme /
   density repaints, is the recommended shape). The inspector already
   guards the other direction — `MeshInspectorPage.set_realized`
   ignores a realization whose `pane_id` is not its view's.
5. **N panes ⇒ N datasets.** Different scopes mean different ghost
   arrays, so panes cannot share one `vtkDataSet`. Use `ShallowCopy`
   (coordinates and connectivity shared, per-pane ghost array added to
   the copy) rather than N deep copies, and measure. This intersects
   the plan's ghost-mask-vs-cell-set-rebuild choice; verify early.
6. **A quad-view legend is proportionally large.** 0090's box is
   content-derived and does not shrink with the pane: a default
   `displacement_z` chip measures 133.5 px, so with the 16 px margin it
   eats ~47 % of a 318 px pane's width — and more for a longer
   component name (a 16-character one reaches 150.9, ~52 %). `font_scale`
   (0090 D4) is the existing knob. Auto-scaling the legend to the pane
   would be a 0090 amendment and is not taken here.
7. **objectNames.** New styled widgets follow 0087 INV-7's PascalCase
   convention — `SessionPaneHost`, `SessionPaneFrame`,
   `SessionPaneHeader`, `SessionPaneTitle`, `SessionPaneStyleButton`,
   `SessionPaneHostEmpty` — with the pane id in a Qt property
   (`paneId`) and the active state as a dynamic property
   (`active="true"`, the `PlotPaneTabRow` pattern), so one QSS block
   styles every pane. (S2's session widgets shipped snake_case names;
   that inconsistency is noted, not reopened.) None of the four
   retired 0088 D6 objectNames may appear.
8. **Tests address panes through the host, not through objectNames** —
   `host.frame(pane_id)` / `host.pane_frames`. Per-pane objectNames
   would break INV-7's one-block-per-component rule.
9. **Inspector pages are never evicted.** `SessionWindow._pages` grows
   only; a page subscribes to the session when constructed
   (`viewers/session/_inspector.py:150`) and is disposed just once, at
   window close (`_window.py:229`). With one pane that is harmless.
   The moment the header's `close` glyph exists, every closed pane
   leaks a live session subscriber refreshing a dead view. S3 evicts
   and disposes the page with the pane.
10. **The reconciler's stale-pane fallback becomes a hazard under N
    panes.** `_reconciler.py:226-242` resolves an unknown or removed
    `pane_id` by clearing it and painting **the first mesh view** —
    correct for one viewport ("the outline re-aims on its next
    selection"), wrong for a host: a closing pane can repaint view 1
    into its dying backend on the removal tick, and a mis-wired pane
    mirrors pane 1 instead of failing criterion 11. Host-owned panes
    need stale → paint nothing. (Caution 3 covers `set_pane`; this is
    a different line.)
11. **First-fit camera is per window, not per pane.** `_window.py:88`
    holds one `_camera_fit` bool, set once at `:205-210`. Kept global,
    pane 2 and beyond boot unframed. It moves onto the pane.

---

### Acceptance criteria (S3)

Lane tags: **[off]** runs in the default CI lane under
`QT_QPA_PLATFORM=offscreen` with injected backends (no GL); **[qt]**
needs a real GL context (`-m qt`, xvfb in CI).

1. **[off]** Boot of a one-pane session: the central widget is
   `SessionPaneHost` containing one root `QSplitter(Horizontal)` with
   one child; `len(host.pane_frames) == 1`; the frame's `paneId`
   equals `session.panes[0].id`.
2. **[off]** The host is a projection: after any sequence of
   `session.add_view()` / `add_plot()` / `remove_pane()` driven from
   **Python only**, `[f.paneId for f in host.pane_frames] ==
   [p.id for p in session.panes]`.
3. **[off]** `T(N)` shape, asserted on the splitter tree for
   N = 1, 2, 3, 4, 5, 6: N = 4 gives a root `Horizontal` splitter with
   two `Vertical` children of two panes each; N = 3 gives columns
   `[p1, p3]` and `[p2]`; N = 5 — the first column-count change, which
   criterion 4 hinges on — gives `[p1, p4] [p2, p5] [p3]`.
4. **[off]** Growth moves panes without rebuilding them. Going
   3 → 4 leaves every existing pane in the same (column, row) cell;
   going 4 → 5 is allowed to move panes between splitters. **In both
   cases, and for every pane, `host.frame(pane_id)` and its backend
   are the SAME objects before and after** — identity, not just
   arithmetic. A host that tears down and rebuilds panes on every Add
   satisfies the cell arithmetic while silently destroying every
   camera (backend state, not session state, so nothing else would
   catch it); this criterion is what refuses it.
4b. **[off]** Ratio survival (A1.2): with `T(2)` ratios dragged to
   70/30, adding a third pane leaves the root splitter's sizes at
   70/30 and resets only the column whose child list grew; going
   4 → 5 resets the root.
5. **[off]** Floors: every pane frame's `minimumWidth() >=
   LAYOUT.pane_min_width` and `minimumHeight() >=
   LAYOUT.pane_min_height`; every splitter reports
   `childrenCollapsible() is False`.
6. **[off]** Add gate: with the host forced to 640 × 671 and four
   panes, the outline's "New mesh view" and "New plot" are disabled and
   their tooltips name the required width; with the host at 960 × 871
   they are enabled. No exception is raised either way.
7. **[off]** The gate does not gate the IR: `session.add_view()` on the
   640 × 671 four-pane host still creates a fifth pane **and** a fifth
   frame.
8. **[off]** Closing the last pane leaves exactly one
   `SessionPaneHostEmpty` card with one action button and zero enabled
   data controls; the inspector shows its nothing-selected hint.
9. **[off]** One active pane at all times: after each of {click a pane,
   click an outline row, add a pane, close the active pane},
   `host.active_pane_id == outline.selected_pane_id()` and the mounted
   inspector page is that pane's.
10. **[off]** Style buttons are per pane and two-way: clicking Outlines
    on pane B sets `session.pane(B).style.outlines` and leaves
    `session.pane(A).style` equal to its previous value; assigning
    `MeshStyle` in Python updates B's button state.
11. **[off]** Pane isolation at the backend: with `RecordingBackend`
    injected per pane, filling pane A's contour slot produces layer ops
    on A's backend only, and every emitted `RealizedPane.pane_id`
    equals its own pane's id.
12. **[off]** One gesture, one realize: filling pane A's contour slot
    in a four-pane session costs exactly one `realize` and one
    `render()` — not four. A theme change costs four (all panes
    repaint).
13. **[off]** Persistence: closing the window writes
    `panes/schema_version == 1` and a `panes/layout` JSON whose `n`
    equals the pane count under
    `QSettings('apeGmsh','ResultsSession')`; nothing is written under
    `QSettings('apeGmsh','ResultsViewer')`;
    `SessionResultsWindow._LAYOUT_SCHEMA_VERSION == 1` (unchanged).
14. **[off]** Restore with a matching `n` reproduces the stored ratios
    (±1 px); restore with a different `n`, a bad version, or corrupt
    JSON falls back to equal ratios without raising and without a
    console message.
15. **[off]** `View → Reset layout` from any arrangement gives the
    0088 D1 dock set **and** `T(N)` with equal ratios.
16. **[off]** Guards: the four retired 0088 D6 objectNames appear
    nowhere in `viewers/**` (criterion 11 of 0088 still green); G-HEX,
    G-SHOUT, G-QSS-SYNC and G-INLINE budgets unchanged or ratcheted
    down; `glyph_names()` contains `outlines`, `nodes`, `gauss`, and
    every style button binds a factory glyph (no text-only fallback).
17. **[qt]** `tests/viewers/test_pane_host_probe.py` stays green
    (7 passed, 0 skipped under xvfb + Mesa).
18. **[qt]** The probe's properties hold on the real `SessionWindow`
    with four mesh panes, not only on the synthetic probe: four
    distinct renderers, independent cameras, one scalar bar per pane,
    a non-flat per-pane screenshot, actors surviving a splitter drag,
    a clean teardown with no leaked interactor, and — the mode the
    tiling law forces — **a live pane surviving a reparent between
    splitters at 4 → 5 with its camera pose and a non-flat frame
    intact**.
19. **[qt]** ADR §11's S3 verify line: two mesh views with different
    slots and different scopes render different pictures — pane A with
    a contour reports one legend, pane B with deform on and no slots
    reports zero (the §5 oracle, per pane).
20. **[qt]** Toolbar camera preset / fit / screenshot act on the
    active pane only: the other panes' camera positions are unchanged.
21. **[qt]** Screenshot review bar (0087 INV-6): 1-pane, 2-pane and
    4-pane captures in `catppuccin_mocha`, `neutral_studio` and
    `paper`, attached to the PR; pane headers legible and unclipped at
    both densities at the 240 px width floor.
22. **Mutation test** (plan standing risk): inverting the `T(N)` fill
    order, dropping the `childrenCollapsible` call, rebuilding panes
    instead of moving them at 4 → 5, or pointing two panes at one
    backend each makes at least one criterion above fail. Recorded in
    the PR.
23. **[off + qt]** `pytest tests/viewers -q` green in **both** lanes,
    restating 0088 D8 criterion 12 — the guard budgets of criterion 16
    are not a substitute for the suite.
24. **[off]** The old window is untouched, which A1.1's no-shell-
    interactor seam promises and nothing else asserts: `results_viewer
    .py` and `viewer_window.py` construction paths are unchanged, and
    the old window's own tests pass unmodified. The seam is additive
    or it is not this amendment's seam.

---

### Open sub-questions left to S3

1. ~~How the shell stops building its own interactor.~~ **Closed in
   A1.1** — the shell builds no central interactor on the session
   path; the seam is an additive, opt-in constructor flag /
   `set_central_widget` the old window never reaches. It was promoted
   out of this list because seven acceptance criteria depend on it.
   What remains for S3 is only the seam's exact spelling.
2. **The per-pane realize guard — spelling only; the gate stands.**
   Criterion 12 is a gate, not a hope: an earlier draft of this list
   also licensed shipping N realizes per tick and "letting criterion
   12 be measured rather than gated", which is a criterion its own
   document authorises failing. The pane-level signature of caution 4
   is buildable — the records are dataclasses, so equality over
   (records, effective instant, style, scope, clips, overlay) is
   cheap — so S3 builds it and criterion 12 holds. What is open is
   only the signature's exact composition. If S3 finds a session
   input the signature genuinely cannot cover, that is an amendment
   to this list, argued on the evidence — not a silent downgrade.
3. **Shared tessellation across panes.** Recommended default:
   `ShallowCopy` + per-pane ghost array (caution 5), verified early
   against the plan's scope-realization choice.
4. **Explicit split direction / pane reordering.** Recommended
   default: not in S3. Both are widenings that need no IR change and
   can land after the loop is proven.
5. **Legend scale in small panes.** Recommended default: leave
   `font_scale` as the user's knob (0090 D4). Revisit only if the quad
   view is reported unusable — auto-scaling is a 0090 amendment.

---

### Consequences

**Positive.** S3's biggest named risk is retired with evidence on both
oracles, and the evidence is a committed regression test, not a
changelog line. The centre becomes N panes with no new dock, no schema
bump and no IR field — 0088's partition and 0098's ontology both
survive intact. Legends, per-pane cameras and per-pane stills come
free from the per-plotter registry. The shape is a pure function of the
pane count, so the outline, the tiling and any future snapshot restore
cannot disagree about what is on screen.

**Negative / accepted.** Adding a pane re-tiles the grid when the
column count changes (N = 2, 5, 10, …), so two panes move once each at
those counts. Hand-dragged splitter ratios are reset by Add, by Close
and by Reset layout. The window size, not a preference, caps the pane
count (4 at 1280 × 800, 9 at 1600 × 1000). A legend occupies about half
the width of a quad-view pane until the operator turns `font_scale`
down. Three new glyphs and four Appendix A rows are owed before the
style buttons can ship. And N live GL contexts is N times the driver
surface — the probe says it works today on both oracles; the probe is
also what will say when it stops.

---

## Amendment 2 (2026-08-18) — the six slotless kinds (S6c)

Append-only. §1–§11, Amendment 1, INV-MESH-1…4, INV-LEGEND-1…5, the
slot catalog and the rejected-alternatives table all stay. This
amendment discharges the Consequences clause that says S6 does not
complete until each shipped kind with no slot has a recorded
disposition.

### A2.1 The six

Six diagram kinds ship today with no §4 slot. §4 named them and
refused them an eighth slot; it did not say what becomes of the code.

| Kind id | Class | Registered at | Dedicated tests |
|---|---|---|---|
| `fiber_section` | `FiberSectionDiagram` | `diagrams/_fiber_section.py:61,68` | `test_fiber_diagram.py` |
| `layer_stack` | `LayerStackDiagram` | `diagrams/_layer_stack.py:59,66` | `test_layer_diagram.py` |
| `isochrone_map` | `IsochroneMapDiagram` | `diagrams/_isochrone_map.py:80,88` | `test_isochrone_diagrams.py`, `test_isochrone_math.py` |
| `isochrone_profile` | `IsochroneProfileDiagram` | `diagrams/_isochrone_profile.py:53,61` | (same) |
| `isochrone_strobe` | `IsochroneStrobeDiagram` | `diagrams/_isochrone_strobe.py:64,72` | (same) |
| `spring_force` | `SpringForceDiagram` | `diagrams/_spring_force.py:60,66` | `test_spring_force.py` |

### A2.2 Decision — internal survival behind the `show_web` hatch

All six survive as internal implementation of the `show_web` hatch.
None is retired. None widens the slot catalog. The catalog stays
closed at seven.

**The catalog was already argued.** §4 rejected an eighth slot and
the alternatives table rejected it again. A slot for any of the six
is a §4 widening argued on evidence at the same review bar as a new
slot — not a side effect of the S6 sweep.

**Retirement would be unargued deletion.** Each of the six is
working, tested code with no replacement in the session ontology and
no recorded complaint. Deleting it at S6 would be the sweep making a
product decision this ADR declined to make.

**Survival is near-free.** The kind registry auto-populates on any
import under `apeGmsh.viewers.diagrams` (`diagrams/_kinds.py:21-27`),
and the hatch keeps a live handle: `WebViewer.director`
(`viewers/web_viewer.py:160`) → `registry.add`
(`diagrams/_registry.py:290`). S6a and S6b keep the six alive by
doing nothing to them; the sweep's import-guard allowlist has to name
the hatch modules regardless, for `render.py`'s tokens and
`studio/_verbs._yield_setup`.

### A2.3 What the disposition costs — stated, not softened

**They lose their authoring UI.** They are reachable today from the
Qt Add-Diagram dialog. After the S6a flip `viewer()` opens the
session window, which has no Add-Diagram dialog, so the only way to
construct one is Python against the `show_web()` director handle — an
internal surface with no compatibility promise (Consequences: the
director "survives past S6 as an implementation detail of that
hatch").

**They were never published, so de-publication costs no page.** Zero
occurrences of the five diagram kinds in `docs/` or in the `apegmsh`
skill (the `spring_force` hits are the results component
`spring_force_0`, not the kind). S6d has nothing to rewrite for them.

**They cannot enter a session.** The §4 catalog is closed and
`restore_snapshot` refuses an unknown slot category, so a picture
made through the hatch is not snapshot state, not pinnable by
`results_pin`, and not renderable by the MCP `render` verb. A hatch
picture is a notebook artifact and nothing else.

### A2.4 Consequences for the sweep

- The five dedicated test files above are **not** retired by S6b.
  They cover code that survives, and S1 does not re-cover them —
  no slot renders any of the six.
- **Corrected during S6b:** those five files are not the whole survivor
  set. Six more tests of these kinds live inside files that are
  otherwise per-kind retirements, and a sweep trusting the list above
  would have deleted them with their hosts:
  `test_diagram_deform_follow.py` (`test_fiber_cloud_follows_deformation`,
  `test_layer_stack_follows_deformation`,
  `test_spring_force_follows_deformation`),
  `test_diagram_recorder_frame.py` (`test_fiber_cloud_uses_recorder_roll`,
  `test_fiber_recorder_z_axes_empty_for_non_ladruno`) and
  `test_no_data_error.py` (`test_spring_force_raises_nodata_when_no_springs`).
  S6b keeps those three files whole rather than splitting them: the
  surviving fraction makes deletion unsafe and the retired fraction
  costs nothing to keep.
- **The mechanism this amendment rests on is now itself tested.**
  `tests/viewers/test_diagram_hatch_survival.py` pins that a bare
  package import registers all six, that one still constructs through
  the real `DiagramRegistry` — the call `WebViewer.director.registry.add`
  makes — and that none of them acquired a §4 slot. Until S6b there was
  no `test_web_viewer.py` and no direct coverage of the hatch at all,
  so A2.2's "survival is near-free" was an untested claim.
- The S6b import guard allowlists the hatch modules. That allowlist
  is the mechanism this disposition rides, so mutation-test it
  against a **near-miss** (a wrong module name in the allowlist),
  not only against an absent one.

### A2.5 Expiry

This disposition is bound to the hatch and does not outlive it.

1. **The web-client ADR retires `show_web` or the director.** All six
   then need a fresh disposition, recorded in *that* ADR before it
   merges. This amendment does not carry them across it.
2. **Any one of the six is asked for in the session window.** That is
   a §4 slot-widening amendment here, argued on the evidence, at the
   same review bar as a new slot — not a reopening of this amendment.

## Amendment 3 (2026-08-19) — the panes are dock panels

Append-only. §1–§11, INV-MESH-1…4, INV-LEGEND-1…5, the slot catalog and
the rejected-alternatives table all stand. Amendment 2 is untouched.

This amendment **reverses Amendment 1's A1.1 decision** — one
`SessionPaneHost` as the window's central widget — and replaces it with
one dock panel per pane. It is **UI, not IR**, exactly as A1 was: no
session record changes, `ResultsSession` / `MeshView` / `PlotView` /
`realize()` and the snapshot are untouched, and layout stays window
chrome in `QSettings`, never in the session.

A1 is reversed on a fact A1 did not have: **the central-widget
arrangement does not work, and cannot be made to work by sizing it.**

### A3.1 Evidence — measured on a real model, not a fixture

Found by opening the `viewer_bench/ssi_frame_wall` bench against a live
run — three panes: a clipped stress contour, a frame moment diagram, a
history plot. Every number below is from that window.

**The pane host is never actually the central widget.**
`SessionWindow.__init__` mounts it (`viewers/session/_window.py:150`) and
it *is* seated; the instant the window is first shown, `centralWidget()`
returns `None`. Traced with an instrumented `QMainWindow`:
`setCentralWidget` is called exactly **once**, with the host, and
`takeCentralWidget` is called **never**. The host object stays alive and
still parented to the window, so it paints at its untouched **640×480**
default *on top of* the docks, while the two side docks split the whole
width (**1254 + 1253** px of 2560). The floating 640×480 pane cluster
users see is that orphan.

**The obvious repair crashes.** Occupying the centre from construction
with a placeholder — so `_build_layout` arranges the docks around a real
centre and `set_central_widget` merely replaces it — makes the host truly
central, and the first show then dies:

```
Windows fatal exception: access violation
  viewer_window.py:1322 in present   ->  showMaximized()
```

and identically on a plain `window.show()`. Reverted.

**The crash is specific to being the central widget.** It is not multiple
GL contexts and it is not size:

| Configuration | Result |
|---|---|
| host as central widget | access violation on first show |
| host orphaned (shipped behaviour) | survives — the 640×480 bug |
| host resized to 1800×900 while orphaned | survives |
| second pane added, re-tiled, resized again | survives |
| host inside one `QDockWidget` | survives show, float, resize to 1400×800 |
| **one dock per pane, 3 panes** | **survives show, float, resize, re-dock, tabify** |

Three timer-based repairs of the central-widget path were tried and all
reverted: a `singleShot(0)` re-seat after show, a deferred `resizeDocks`,
and the placeholder centre. The first two measure perfectly on a
fixed-size `show()` and wreck the real *maximized* window; the third
crashes. **A Qt layout fix must be verified on the path the user runs —
maximized, real window — not on a fixed-size `show()`.** That mistake was
made twice during this investigation and reported as success once.

### A3.2 Decision

**Each pane is its own `QDockWidget`** — movable, floatable, closable —
placed with `splitDockWidget` at creation. The window keeps ADR 0088 D1's
partition (outline left, panes in the middle, inspector right, scrubber
bottom), but the middle is dock panels, not a central viewport.

What this buys beyond not crashing: a pane can be dragged out, floated,
resized on its own, tabbed onto another pane and restored — all from Qt,
none of it hand-written. A1.2's `T(N)` exists to answer "where does pane
N go" with no user input; Qt's dock layout answers the same question and
then lets the user override it, which is what a viewer should do.

**A1.2's tiling law, `layout_ratios` / `apply_ratios` / `reset_tiling`
and the splitter host retire with it.** A1.3's pane frame survives
unchanged — it becomes the dock's widget. A1.4's floors survive as dock
minimum sizes. A1.5's active-pane rule survives, expressed as dock focus.

### A3.3 Persistence — who owns the arrangement

Two systems now describe the panes, and they must not fight.

**The session is authoritative for existence; the saved arrangement is
advisory for placement.** Concretely:

1. `ResultsSession.panes` decides which panes exist. Nothing in
   `QSettings` may create or destroy a pane.
2. Each pane dock carries a stable `objectName` derived from the pane id,
   so `saveState` / `restoreState` can address it.
3. On restore, a saved entry naming a pane that no longer exists is
   **dropped**; a pane with no saved entry takes **default placement**.
   Neither is an error and neither prompts.

Qt implements exactly this, but only through
`QMainWindow.restoreDockWidget`, not through a second `restoreState`:
the window restores its state before any pane dock exists, each pane
dock is then seated at default placement and claims its own entry out
of the pending state. Re-applying the whole state after the fact was
tried and measured — it leaves the entire Left area, outline included,
unplaced at negative coordinates.

This keeps §1's law intact — layout is chrome, the snapshot is the
document — and makes the mismatch case (a session snapshot restored
beside a stale window layout) decidable instead of undefined.

### A3.4 Geometry lessons that are contract, not folklore

All three were measured, and all three are silent when wrong:

- **`resizeDocks` across dock areas needs two passes, one event-loop turn
  apart.** The Left chain (outline, panes) settles on the first pass and
  the Right one (inspector) only on the second; two calls inside one turn
  move nothing.
- **`resizeDocks(..., Vertical)` does not move the scrubber.** It sat at
  730 px through every variant. A maximum-height ceiling does.
- **The inspector needs a width floor.** Its slot rows and Add buttons
  measure ~375 px; without `setMinimumWidth` the settle leaves it at ~352
  and clips them.
- **A dock derives its minimum from its widget.** A 240 x 200 pane frame
  makes its dock measure 240 x 224 with no `setMinimum*` call on the dock
  at all — so A1.4's floors reach the dock through the FRAME, and
  restating them on the dock can only ever *lower* the real floor (it
  took the height from 224 back to 200). The floors stay on the frame.
- **View -> Reset layout has to re-apply the extents.** The default dock
  state the shell restores predates the panes, so re-seating them alone
  lands every pane on its A1.4 minimum with the middle column half empty
  (measured: 240 + 301 px of 1871). The same two-pass schedule as the
  boot path fixes it.

And one trap that cost two measurement rounds: `ResultsWindow._build_layout`
imports `QtCore` **locally**, so code factored out of it loses the name —
and the `resizeDocks` block sits inside `except Exception: pass`, which
turns the resulting `NameError` into a silent no-op.

### A3.5 Consequences

- `SessionResultsWindow._LAYOUT_SCHEMA_VERSION` goes **1 → 2**: the dock
  set changed structurally, so v1 state must not half-apply. The retired
  objectName discipline applies — `dock_results_panes`, the intermediate
  single-panel spike, must not be reused for different content.
- `tests/viewers/test_pane_host.py` encodes the central-widget contract
  (`centralWidget() is window.host`) and the `T(N)` splitter geometry; it
  is rewritten with this amendment, not patched around.
  `test_session_window_schema_numbering_restarts_at_1` asserts schema 1
  and moves to 2.
- The S3 acceptance criteria that name splitters or `T(N)` are restated
  against dock placement. The criteria about *what a pane draws* are
  untouched — this amendment never reaches the IR.
- ADR 0088 D1's "viewport keeps ≥ 50 % of the width" survives as a
  property of the panes' dock column and is now measurable: on a 2560 px
  window the panes hold 1870 px against outline 253 and inspector 380.

### A3.6 Alternatives rejected

| Rejected | Why |
|---|---|
| Keep the central widget, fix the sizing | Not a sizing bug — the centre is dropped on first show, and forcing it in crashes |
| One dock holding the splitter host | A movable *container*, not movable views; buys nothing over the centre except not crashing |
| One render window with N viewports | Would dodge multi-context GL entirely, but the evidence says multi-context is not the fault (two panes resize fine) — a rewrite of the pane host and the backend seam for a problem we do not have |
| MDI subwindows | Still rejected on A1's original UX grounds; docks are not MDI and give tabbing and persistence for free |
| Leave it broken and document it | The window is the product; a 640×480 cluster painted over the docks is not a viewer |

## Amendment 4 (2026-08-19) — a time change stops rebuilding the pane

Append-only. §1–§11, INV-MESH-1…4, INV-LEGEND-1…5, the slot catalog and
the rejected-alternatives table all stand. Amendments 1–3 are untouched,
and this one never reaches the IR: no session record changes, `realize()`
still exists and still produces exactly what it produced.

This amendment **lifts the one-shot restriction** `_reconciler.py` froze
in S2 — "update-in-place by key is deliberately NOT the v1 protocol" —
for exactly one case: the instant moved and nothing else did. That
restriction was the right call with no evidence; there is evidence now.

### A4.1 Evidence — measured on the bench, with a harness that stays

`internal_docs/profiling/results_session/profile_session.py` drives a
deterministic gesture workload against `ssi_frame_wall`,
`c001_baseline_small`. Every number below is from it, `--no-profile`
(cProfile inflates wall clock ~1.4x):

| gesture | mean ms | p50 | p95 |
|---|---:|---:|---:|
| **scrub** | **405** | 401 | 425 |
| pane add+close | 336 | 332 | 359 |
| slot fill/clear | 166 | 118 | 285 |
| scope flip | 57 | 27 | 173 |
| camera orbit | 18 | 16 | 37 |

The scrubber defaults to 30 fps — 33 ms. Scrub is **12x over budget**,
and it is not the GPU: VTK `Render` is 7 ms/step and an orbit is 18 ms.

Pane count is not the driver either. One pane is already most of it:

| panes | scrub mean ms |
|---|---:|
| 1 (contour + deform + clip) | 284 |
| 2 (+ line diagram) | 399 |
| 3 (+ history plot) | 408 |

The plot pane costs **9 ms** — `PlotReconciler`'s cheap cursor path
working exactly as designed, and the existence proof for this amendment.

Attribution, 10 steps on one pane with a Gauss contour:

| | cum s | ncalls | |
|---|---:|---:|---|
| `_reconciler._flush` | 3.86 | 20 | the pump |
| ↳ `realize_pane` / `_realize_mesh` | 3.27 | 10 | once per step — correct |
| ↳↳ `extrapolate_gauss_slab_to_nodes` | 2.43 | **20** | twice per step |
| ↳↳↳ `extrapolate_gauss_slab_per_element` | 1.61 | 20 | 164 020 `np.broadcast_to` |
| `_contour.attach` | 1.60 | 10 | a full re-attach every step |
| `_contour.update_to_step` | 1.25 | 10 | |
| `_reconciler._teardown` | 0.47 | 10 | |
| VTK `Render` | 0.74 | 80 | 7 ms/step |

**Two of the three causes are plain bugs, not design.**

**Bug A — the Gauss accumulator is dead on every unscoped view.**
`_specs._scope_ids` returns `None` for an unscoped view, meaning *all
elements*. `_contour._build_gauss_state` reads that `None` as *nothing to
build* and returns early. So the default whole-mesh pane gets no
`GaussToNodeAccumulator` — the per-element Python loop above — and no
`_gauss_gp_mask`, which makes `_visual_gauss_step_subslab` bail on the
same guard and re-read HDF5 every step instead of using the RAM slab.
Both halves of the optimisation that commit was written to deliver are
inert on the common path. `tests/test_gauss_extrapolation.py` covers the
accumulator in isolation; nothing asserted that a contour ever builds
one. **This is in `viewers/diagrams/_contour.py`, shared with the old
viewer — an unscoped Gauss contour is slow there too.**

**Bug B — a fresh `VisualDataStore` per flush.** `_realize_mesh`
constructs one, so a cache whose stated purpose is 30 fps scrubbing from
RAM is empty on every step and dies with the call — and because
`ContourDiagram.attach` pre-warms it, every flush pays a full
`(T, N_gp)` HDF5 read and throws it away. The old viewer owns exactly one
on the director for the whole session. N panes currently mean N stores of
the same slab.

**Cause C — the design.** `_pane_signature` is one flat tuple with the
instant inside it, so any step change fails equality; `_flush` then tears
down every layer and scalar bar and re-realizes, which rebuilds
`build_fem_scene`, `ViewerData.from_fem`, the scoped grid, and a
brand-new diagram per slot whose `attach()` redoes submesh extraction,
`isin` and the accumulator build — *before* calling `update_to_step`.

### A4.2 Decision

**The pane signature splits in two, and a cursor-only change re-steps
instead of re-realizing.**

    signature = (structure, cursor)
    structure = everything the picture is built FROM
    cursor    = session.effective_instant(pane)

* `structure` equal, `cursor` moved **within one stage**, not forced, and
  the pane re-steppable → drive `update_to_step` through
  `RealizedPane.diagrams`, re-pose in place, re-point the realize-owned
  layers from row maps realize recorded, refresh the pick targets.
  No teardown, no `realize_pane`, no actor churn.
* Anything else → today's one-shot path, unchanged.

This is not a new idea in this codebase. It is what `PlotReconciler`
already does for plots (`_chart.py`), what the old viewer's `pump_step`
did for every diagram, and what `RealizedPane.diagrams` was added to
allow — `_reconciler.py` names this exact fix as owed work.

`deform` stays in `structure`: a scale change is rare and a full realize
is the conservative answer. The selection term stays in `structure` too,
so the S4-1 selection-repaint defect stays fixed.

### A4.3 The invariants the fast path must preserve

Each one names the oracle that holds it. A fast path without these is a
stale-picture generator, which is precisely what the one-shot rule was
protecting against.

1. **Parity.** The layers the fast path leaves behind are identical to
   what a forced full realize produces at the same instant. Oracle: scrub
   several steps, snapshot every layer's points and fields, then
   `schedule_forced()` at the same instant and compare. The oracle is
   itself mutation-tested — stub the re-step to a no-op and the parity
   body must fail.
2. **Criterion 12 both halves.** A step-only tick calls `realize_pane`
   **zero** times; a theme change still costs every pane a full realize
   (`_forced` bypasses the fast branch entirely).
3. **One render per gesture.** Structural, not asserted: both branches
   fall through to the single existing `render()` call.
4. **No scalar-bar churn.** `backend.scalar_bars` is identical before and
   after a scrub — the contour's clim comes from the store's
   whole-history limits and is fixed at attach, so tearing bars down per
   step was never buying correctness.
5. **Pick targets stay pose-current.** `realized.targets` is what
   `PanePick` and the window read; after a scrub with a pose it must
   equal the posed coordinates.

### A4.4 When the fast path refuses

Refusing is free — it costs today's time and today's behaviour. It
refuses when:

* realize could not record a row map for a layer it owns;
* the **stage** changed (the cached arrays are a function of the stage —
  the same limit `PlotReconciler._same_stage` already enforces);
* the flush is `_forced` (theme / density);
* an occupied slot's kind re-fits its LUT to the displayed values per
  step (`gauss_marker`, `vector_glyph`, `principal_glyph`, `sand`, with
  no explicit `clim`) — reproducing that means re-registering the bar,
  which this amendment does not take on. `contour` is deliberately not in
  that set, so the hot case is unaffected;
* the re-step raises — it reports and **falls through to the full realize
  in the same flush**, never leaving a half-stepped picture.

### A4.5 Consequences

- `_reconciler.py`'s "update-in-place is not the v1 protocol" docstring is
  restated: it is the protocol for a cursor-only change and nothing else.
- **Bug A's fix changes the old viewer too**, and for the better — it is
  the same `ContourDiagram`. It is covered by a bit-identical parity test
  against the fallback path rather than by assertion.
- **Bug B's fix holds cached slabs for the life of the `Results`**, which
  is the old viewer's behaviour; the `APEGMSH_VIEWER_CACHE_BYTES` LRU
  already exists for the models where that matters.
- `NodeTargets` / `GaussTargets` gain the row maps realize already
  computes and discarded; `RealizedPane` gains one `restep` carrier. All
  additive.
- **Achieved** (same bench, same harness): one pane **294 -> 18 ms**,
  three panes **405 -> 74 ms** — 16x and 5.5x, under the 33 ms budget.
  The Gauss and nodal cases converge at 18 ms because with no re-attach
  per step the extrapolation is paid once and never again. The
  structural gestures are deliberately unchanged (`slot` 148, `scope`
  60, `pane` 325): they take the full path, as A4.2 says they should.
- One visible consequence of ordering the store warm-up before the clim:
  a Gauss contour on the session path now carries the WHOLE-HISTORY
  colour scale rather than one fitted to step 0. That is what the old
  viewer does and what S1-A P10 intended, and it is what makes a scrub
  comparable frame to frame — but it does change how an existing pane
  looks, and a single step reads flatter against the wider scale.
- The profiling harness is part of the deliverable, not scaffolding:
  `internal_docs/profiling/results_session/` keeps the workload and the
  measured trajectory so the next regression is caught by a number rather
  than by feel.

### A4.6 Alternatives rejected

| Rejected | Why |
|---|---|
| Fix only Bugs A and B | Gets ~294 → ~90-120 ms. Real, cheap, and still 3x over budget: the per-step teardown-and-rebuild is the floor |
| Cache `realize_pane` on the signature | Memoizes the wrong thing — the output is layers already pushed to a backend, and the cost is the rebuild, not the diffing |
| Move the whole pane onto the old `ResultsDirector` | Reinstates the director the session was designed to remove (§1); the parts worth having are the store and `update_to_step`, and both are reachable without it |
| Throttle the scrubber to what the renderer can do | Hides the defect and makes playback lie about time |
| One shared diagram set across panes | Panes are independent by construction (criterion 11); sharing them re-creates the coupling N panes exist to avoid |

## Amendment 5 (2026-08-19) — the colour scale is direct-manipulable again

Append-only. §1–§11, INV-MESH-1…4, INV-LEGEND-1…5, the slot catalog and
the rejected-alternatives table all stand. Amendments 1–4 are untouched.

Unlike Amendment 4, **this one does reach the IR**: §5 keeps legend
*existence* derived, and adds a place for legend *placement* to live.

### A5.1 Evidence — the gesture was never carried into the session window

Dragging and resizing a colour scale is ADR 0081 L2, and it works: the
gesture lives in `viewers/core/_legend_interactor.py`, deliberately in
the interactor event stream at priority 12 rather than in a
`vtkScalarBarWidget` (which could never receive a click under the pick
engine at priority 10). `pyvista_qt.add_scalar_bar` still says so:
`interactive=False` is load-bearing, and "drag and resize live in the
controller's own interactor instead."

`install_legend_interactor` has **exactly one caller in the codebase** —
`viewers/results_viewer.py:4317`, the old window. S6a flipped `viewer()`
to `session().show()`, so every user now lands in a window that never
installs it. This was not blanket neglect: the session pane does wire
navigation (`session/_pane.py::apply_pane_navigation`) and picking
(`session/_pick.PanePick`). The legend gesture was simply not in the set
that was ported.

Measured against the bench (`ssi_frame_wall`, `c001_baseline_small`, one
pane, `Contour("stress_zz")`, viewport 1866x729):

| question | result |
|---|---|
| Does `install_legend_interactor` attach to a pane's interactor? | **yes** — `LedgerBackend.__getattr__` forwards `.plotter` |
| Does a synthesized press-move-release move the bar? | **yes** — anchor `(0.823, 0.393)` to `(0.758, 0.229)` |
| Does the moved anchor survive a full realize? | **no** |

So the gesture is not broken and the plumbing is not missing. One call
is missing — and that call alone would still not be a fix.

### A5.2 Why installing it is not the fix

The same probe forced a full realize (a `deform` change, i.e. what any
slot edit does) and read the controller back:

* the pane's `legend_controller` is a **different object** — every full
  realize runs `_realize_legends`, which does `LegendController(backend)`
  unconditionally (`viewers/session/_realize.py:1493`);
* the anchor came back as `(0.943, 0.394)` — **neither the dragged
  position nor the position before the drag.** The new controller lays
  out from scratch, so a dragged bar does not merely snap back, it jumps
  to a third place;
* a resize would be reverted for a second, independent reason:
  `_realize_legends` seeds `font_scale` from `style.scalar_bar_scale`, so
  the style record overwrites the gesture on every realize.

And there is nowhere to persist any of it. §5 defines legends as derived,
and the only per-legend state the view or the snapshot carries is
`legend_hidden` per field (`results/session/_snapshot.py:208`), so a
placement would not survive save/restore either.

Three layers, one finding — the same shape as Amendment 3's pane host,
where installing the obvious fix in isolation made things worse.

### A5.3 Decision — placement is session state, existence stays derived

1. **`MeshView` gains per-field legend placement**, beside
   `legend_hidden`: for a field, an optional `(anchor, font_scale)`.
   Absent means "laid out automatically", which is today's behaviour
   and stays the default.

   **Not `extent`.** The draft of this amendment said
   `(anchor, extent, font_scale)`, which contradicts A5.4 one paragraph
   later: `extent` is *resolved* by the layout from the legend's text
   and its font scale, so storing it would be storing a pixel box under
   another name — the thing A5.7 rejects. Corrected during
   implementation.

   **And not a field of `Legend`.** `Legend` records go into the
   reconciler's structure signature (`_pane_signature` includes
   `pane.legends()`), so a placement carried there would make every
   mouse-move of a drag compare unequal and cost a full realize — the
   opposite of criterion 9, and about 150 ms per frame on the bench.
   Realize reads placement through `view.legend_placement(field)`
   instead. This is a constraint on the design, not an implementation
   detail: any future per-legend state has the same choice to make.
2. **`_realize_legends` seeds the controller from that record**, not from
   `style.scalar_bar_scale`, whenever a placement exists for the field.
   The style value remains the seed when it does not.
3. **The pane installs the legend interactor**, bound to the controller
   the reconciler just adopted, and re-binds it whenever realize swaps
   the controller. Re-binding on swap is required, not optional: the
   controller object is not stable across a realize (A5.2).
4. **The interactor's mutators write back to the session view.**
   `set_anchor` / `set_font_scale` / `redock` already funnel every gesture
   through the controller (ADR 0081 L2); the controller notifies, and the
   pane records the result on the view. A dragged bar is session state,
   so it survives realize, theme changes and snapshot/restore.
5. **The snapshot carries placement** under the same rule `legend_hidden`
   already uses: a placement for a field this view no longer causes a
   legend for is **dropped loudly on restore**, never resurrected.

`redock` clears the record for that field, which is what returns a bar to
the automatic stack.

### A5.4 What this deliberately does not change

* **Legends stay derived.** `legends = f(occupied colour-mapped slots)`
  is unchanged; INV-LEGEND-1 (a scale belongs to a painted field) is
  unchanged. Only *where a legend sits* becomes remembered state. You
  still cannot author a legend that no slot causes.
* **The automatic layout stays the default and stays authoritative for
  anything unplaced.** ADR 0081 Part 3 sizes a legend from its text; a
  resize drives `font_scale` and lets the box follow, which is why a bar
  can never be dragged to a size its labels do not fit. That property is
  preserved by storing `font_scale` rather than a pixel box.
* **`interactive=False` stays.** Nothing here revives the
  `vtkScalarBarWidget`; the priority-12 interactor remains the mechanism.

### A5.5 The two sibling gestures, named not solved

`install_clip_gizmo_interactor` and `install_scope_gizmo_interactor` have
the identical single-caller problem — both are called only from
`results_viewer.py`, so the clip-plane and scope gizmos are also
unreachable in the session window. They are **not** in this amendment:
the clip gizmo writes to `ViewClip`, which the session record already
owns, so it is a wiring job without A5.3's state question. Recorded here
so the set is known to be three, and so the next person does not
rediscover it one gizmo at a time.

### A5.6 Acceptance criteria

1. With a contour slot filled, the pane installs a legend interactor;
   a headless/offscreen backend installs none and does not raise.
2. A synthesized press-move-release over the bar changes the view's
   recorded placement for that field — not just the controller's.
3. After a full realize (slot edit, scope flip, theme change), the bar is
   still where it was dropped. **Mutation test:** removing the re-bind in
   A5.3(3) must fail this, and removing the seed in A5.3(2) must fail it
   differently — the bar jumps to a fresh layout, as measured in A5.2.
4. A resize survives a realize, and cannot be driven below the size the
   labels fit.
5. `redock` clears the record and the bar rejoins the automatic stack.
6. Snapshot round-trip preserves placement; a placement for a field with
   no live legend is dropped with the same loud notice `legend_hidden`
   uses, and never resurrects.
7. A pane with no legend, and a legend hidden via `set_legend_hidden`,
   install and record nothing.
8. Two panes, each with its own bar: dragging one must not move the
   other. Legend placement is per view, not per session.
9. Criterion 12 still holds — installing the interactor must not cost a
   realize, and a drag must not trigger one.

### A5.7 Alternatives rejected

| Rejected | Why |
|---|---|
| Just call `install_legend_interactor` in the pane | Measured: the drag lands, then the next realize throws it away and the bar jumps to a third position (A5.2). A gesture that visibly un-does itself is worse than one that is absent |
| Make `_realize_legends` reuse one controller per pane | Attractive, and it would fix the object churn — but the controller is bound to layer handles that realize legitimately replaces. Reuse means teaching it to re-bind, which is the same work as re-seeding with none of the persistence |
| Store a pixel box instead of `font_scale` | Breaks ADR 0081 Part 3's fit guarantee: a box that does not derive from the text can be dragged smaller than its labels |
| Put placement in `MeshStyle` next to `scalar_bar_scale` | Style is per slot; a legend can be shared by a contour and a Gauss slot of the same field (§5). Placement keyed by field, on the view, is the only key that matches what a legend *is* |
| Revive `interactive=True` | The reason it never worked is unchanged: the widget observes at priority 0.5, the pick engine aborts LMB at 10 |
| Leave it to the old viewer | S6a made `session().show()` the only window a user gets; "use the retired viewer for this" is not an answer |
