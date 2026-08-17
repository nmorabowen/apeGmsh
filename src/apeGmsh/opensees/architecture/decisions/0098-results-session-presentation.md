# ADR 0098 — Results presentation is a `ResultsSession` of views, not geometries of diagrams

**Status:** Proposed (2026-08-16) — design ratified in the owner
workshop the same day; record revised the same day after adversarial
review. Implementation has not started. Qt layout of the pane host
is a named follow-up (see *Open question 1*); the session IR does
not wait on it.

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
   as amended above. The answer is recorded as Amendment 1 of this
   ADR before S3 starts; it does not block S0–S2.
2. **Averaging region.** v1 averaged contour is global on the
   visible cell set. Averaging only within a material / PG is a
   later widening of the contour slot.
3. **Web client.** Same session, later ADR. The director survives
   only as the `show_web` hatch’s internal implementation until
   then (Consequences); it does not survive as an ontology or a
   public surface.
