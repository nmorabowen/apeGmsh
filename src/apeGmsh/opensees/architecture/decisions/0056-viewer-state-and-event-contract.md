# ADR 0056 — Viewer state & event contract: single owners, owner-fired events, reconciler-only artifact writes

**Status:** Proposed (2026-06-10). Completes the viewer-discipline
arc: ADR 0014/0026 disciplined the viewer's **read** path, ADR 0042
the **render** path, ADR 0045/0047 the **pick/selection** path. All
of those govern *what* flows across a seam. None governs *when and
why the picture changes* — state ownership and re-render triggering
have no contract, and every chronic viewer bug class of the last two
months (diagrams toggling inconsistently across UI surfaces, hidden
elements retaining ghost nodes, runtime toggles lost on panel
rebuild) traces to that gap. The surgical pre-slice shipped as
PR #593 (V0 below); this ADR ratifies the contract those fixes
conform to and sequences the rest.

## Context

### The dispatcher exists, is well-designed, and is optional

The results viewer already has the right machine:
`viewers/diagrams/_dispatch.py` defines a single event matrix over
four primitives (STEP / DEFORM / GATE / RENDER), two subscriber
lanes (synchronous RENDER, Qt-deferred coalescing UI), one coalesced
`plotter.render()` per fire, `session_batch()` bulk suppression, and
a session action log capturing every dispatch for replay. Its module
docstring states the intended contract verbatim:

> "Every UI gesture / observer / shortcut funnels through
> `Dispatcher.fire(event_kind, ...)` … This is the only place those
> four primitives may run."

That sentence was prose, not a contract. The June 2026 audit (three
exploration passes + line-level verification, leading to PR #593)
found the violations exactly where prose-only rules predict them:

1. **The composition gate was a silent no-op for every diagram.**
   `pump_gate` flipped `d._actors` directly — but the ADR 0042 R-B
   migration moved all 11 diagram kinds onto backend layer handles,
   so `_actors` was empty for all of them. Composition-scoped
   show/hide never reached the screen, and nothing failed.
2. **The outline tree's eye icons bypassed the dispatcher.** All
   three row types (layer / composition / geometry) mutated
   visibility through `registry.set_visible(...)` and then called
   `plotter.render()` directly — no `LAYER_VISIBILITY_CHANGED`, no
   gate re-run, no render coalescing, no session-log entry — while
   the settings-tab checkbox for the *same state* fired the event.
   Two UI surfaces, one boolean, two propagation behaviours.
3. **Write-only runtime state.** The settings tab restored its
   "show undeformed" checkbox from `_runtime_show_undeformed`, an
   attribute no code ever wrote; the actual toggle went straight to
   a backend handle, so a plain `set_visible(True)`, a gate pump, or
   a stage-change re-attach resurrected a ghost the user had turned
   off.
4. **Silent degraded fallbacks.** The mesh-scene per-entity
   `getNodes` pass swallowed exceptions (`except: pass`); the
   node-cloud rebuild then deliberately no-ops without ownership
   data — a locked, reasonable fallback rendered indistinguishable
   from a visibility bug because neither end said anything.

Equally instructive is what the audit did **not** find broken: the
Auto-Apply debounce path was reported as a dispatcher bypass by an
exploration agent and turned out, on source verification, to fire
`DIAGRAM_MODIFIED` correctly (`_flush_auto_commits` runs each card's
commit closure, which dispatches). Where the dispatcher is wired,
the machine works. The failure mode is *bypass*, never *mechanism*.

### One boolean, five homes

A single diagram's visibility currently lives in: `Diagram._visible`
(the intent flag), the settings-tab card checkbox, the outline-tree
eye state, per-actor `SetVisibility` flags, and backend layer-handle
`visible` flags. Nothing makes these agree except the discipline of
every call site — which is exactly what failed above. The mesh and
model viewers have their own parallel propagation mechanisms
(`VisibilityManager.on_changed`, `OverlayVisibilityModel.on_changed`
plain-callback lists, the partially-deployed `_active_objects.py`
Qt-signal hub from Plan 04) with the same unstated rules.

### Why this is an ADR, not a refactor

Two reasons. First, the contract spans subsystems (diagrams, UI
panels, outline trees, three viewers, the dispatcher) and future
code must conform — that is an architecture rule, not a cleanup.
Second, this codebase is built by multiple agents in parallel, and
the audit demonstrates that prose conventions do not survive
multiagent development; the dispatcher's own docstring already
claimed the rule this ADR enforces. The project's proven remedy is
the machine-checked guard test (`test_viewers_pure_h5_consumer.py`,
`test_diagrams_pure_no_pyvista.py`, `test_scene_ir_pure.py`). This
ADR extends that pattern to event flow.

## Decision

Six parts. Parts 1–4 are the contract; Part 5 makes it
machine-enforced; Part 6 extends it beyond the results viewer.

### Part 1 — Every piece of view state has exactly one owner

View state is classified into three kinds:

- **Intent state** — what the user asked for: per-diagram visibility
  (`Diagram._visible`), runtime style overrides (scale,
  show-undeformed), step/stage position, active geometry /
  composition, hidden entity/cell sets, overlay pattern selections,
  color modes, opacities. **Each field has exactly one owning
  object** (the diagram for its own runtime overrides, the Director
  for step/stage, Geometry/CompositionManager for the tree,
  `ElementVisibility` for cell masks, `OverlayVisibilityModel` for
  mesh overlay sets, …). The owner is where reads come from and
  where writes go.
- **Derived (effective) state** — anything computable from intent:
  the gate result (`effective = is_visible AND in_active_comp`),
  actor/handle visibility flags, ghost-cell masks. Derived state is
  **never stored authoritatively** — it is recomputed and pushed by
  the reconciler (Part 4). PR #593's
  `Diagram.apply_effective_visibility(effective)` is the canonical
  shape: it pushes effective visibility through the subclass's
  artifact path while explicitly preserving the intent flag,
  because the gate *reads* `is_visible` and writing the gated value
  back would corrupt its own next input.
- **Widget state** — checkboxes, sliders, tree rows. Widgets are
  **projections**: they render owner state and forward gestures to
  owner mutators. A widget (or a write-only attribute beside it) is
  never the sole holder of view state; every widget must be able to
  rebuild itself from owners alone. (The `_runtime_show_undeformed`
  incident is the counterexample this rule exists to kill.)

Selection state (ADR 0045's `SelectionState` + `SelectionLog`) is
intent state under this taxonomy with `SelectionState` as its owner;
it keeps its own observer list and undo log today. Whether its
observers migrate onto the dispatcher's UI lane is decided per
viewer at V3/V4 (open question 3) — the ownership classification is
ratified here, the transport is not.

### Part 2 — Owners fire events; call sites only call mutators

Today the *call site* is responsible for firing the dispatcher after
mutating (`registry.set_visible(d, v)` then `disp.fire(...)`), which
is precisely the forgettable step that produced the outline bypass.
The contract inverts it:

- A mutator on the owning object both applies the change **and**
  fires the matching dispatcher event. `DiagramRegistry.set_visible`
  fires `LAYER_VISIBILITY_CHANGED`; Geometry/CompositionManager
  mutators keep firing their granular typed events (already the
  case); `ElementVisibility` keeps firing
  `ELEMENT_VISIBILITY_CHANGED` (already the case).
- UI handlers, shortcuts, and pick handlers call mutators and do
  nothing else. They never fire pump-bearing events for state they
  don't own, and never touch render artifacts.
- Idempotent-skip stays legal **per call**: a mutator that detects a
  no-op write (value unchanged) may skip the fire — the
  `OverlayVisibilityModel` setter pattern. It is per-call only; a
  mutator never tries to coalesce across calls (that is the batch's
  job, below).
- **Bulk gestures batch.** Owner-fired events turn an N-layer loop
  (the outline's composition/geometry eye cascade walks
  `registry.set_visible` over every child layer) into N synchronous
  pump+render sequences where today the call site fires once after
  the loop. Any gesture that mutates more than one owner field
  therefore wraps in a dispatcher batch. `session_batch()` (exists)
  replays a *full* pump on exit — correct for session restore,
  heavier than needed for a visibility cascade — so the dispatcher
  gains a sibling `gesture_batch()`: suppress inside the block,
  then run the **matrix-row union of the suppressed kinds** once,
  plus one render. A composition toggle over N layers is then N
  intent writes, one gate pump, one render — the same cost as
  today, with the fires owner-side.

You can't forget what you don't have to remember: once owners fire,
the bypass class is unrepresentable at call sites — and once bulk
call sites batch, the fire-per-mutation cost is one pump per
gesture, not per field.

### Part 3 — The dispatcher always exists

The Dispatcher is currently constructed inside `ResultsViewer.show()`
and injected as `director.dispatcher`, so every consumer defends with
`getattr(director, "dispatcher", None)` and grows a fallback branch
(the outline's raw-render fallback — kept in PR #593 only because
headless contexts have no dispatcher). Fallback branches are
mini-bypasses waiting to drift.

The Director constructs a Dispatcher at `__init__` with no-op pumps
and no defer function; the viewer **rebinds** real pumps + render +
defer at `show()`. Mechanically: the `Dispatcher` constructor's pump
arguments become optional, defaulting to no-op callables defined in
`_dispatch.py` itself (so there is no import cycle to worry about —
`_director.py` and `_dispatch.py` are same-package siblings and
`_dispatch.py` imports nothing from the director side), and a
`bind(pump_step=…, pump_deform=…, pump_gate=…, pump_restack=…,
render=…, defer_fn=…)` method rebinds them. Consequences:

- `director.dispatcher` is never `None`; the seven
  `getattr(director, "dispatcher", None)` defenses measured at
  ratification time (results_viewer ×1, settings tab ×5, outline ×1)
  and the two raw-render fallback branches (outline, settings tab)
  are deleted.
- Headless and unit-test contexts exercise the *same* event path as
  the live viewer (asserting on fired events instead of monkeypatched
  fallbacks), which is what makes the Part 5 guards safe to enforce.

### Part 4 — Only the reconciler touches render artifacts

Each viewer has **designated reconciler code** — the pump callables
bound to its dispatcher plus the `RenderBackend` implementations —
and that code is the **only** code that translates view state into
artifact mutations (actor flags, handle visibility, point arrays,
ghost masks, actor add/remove); the dispatcher is the only caller of
`render()`. In the results viewer the reconciler is the four
existing pumps. In the mesh viewer, `VisibilityManager` today plays
*both* roles — its mutators (`hide` / `isolate` / `set_hidden`) own
the `_hidden` state **and** immediately call `_rebuild_actors()`,
an artifact write inside the owner. That is not a contradiction to
paper over but precisely the coupling V3 unwinds: the mutators keep
the state and fire the event; `_rebuild_actors()` becomes the mesh
dispatcher's pump for that event (same code, invoked one hop later).
Explicitly preserved:

- The in-place fast paths stay (vtkGhostType recompose, point-array
  `update_layer`, per-layer submesh sync). The reconciler is
  diff-shaped, never rebuild-the-world; the existing perf gates
  (`test_dispatcher_perf.py`, `test_inplace_mutation.py`) remain the
  arbiter.
- Diagrams' own artifact methods (`set_visible` /
  `apply_effective_visibility` / `update_to_step` /
  `sync_substrate_points`) are reconciler *callees* — they are
  invoked by pumps and owner mutators, not directly by UI code.
- One coalesced render per fire (already the dispatcher's behaviour)
  becomes the only render path in interactive flow.

### Part 5 — The contract is machine-enforced (AST guards)

A new guard test, `tests/viewers/test_viewer_state_contract.py`,
in the established AST-guard pattern:

- **G-RENDER** — in `src/apeGmsh/viewers/ui/**`: no call expression
  whose attribute is `render` (catches `plotter.render()` and
  `self._plotter.render()` alike).
- **G-ARTIFACT** — in `src/apeGmsh/viewers/ui/**`: no calls to
  `SetVisibility`, `set_layer_visible`, `SetPickable`, or
  `add_mesh` / `remove_actor`.
- **G-IMPORT** — `src/apeGmsh/viewers/ui/**` imports neither
  `apeGmsh.viewers.backends` nor `pyvista` / `vtk*` directly.

Each guard carries an explicit, enumerated allowlist for
transitional violations, burned down per adoption slice; adding an
allowlist entry requires a comment citing this ADR and a reason.
The baseline was measured at ratification time and is tractable:
**seven** `render()` calls in `ui/` (five in the `viewer_window.py`
window host — control-layer renders for theme/layout changes, the
durable allowlist candidates — plus the two transitional fallback
branches V1 deletes), **zero** artifact-flag calls, **one** direct
`pyvista` import (`viewer_window.py`).

The initial `ui/` scope is deliberate: `core/` contains reconciler
helpers (visibility managers, pick engines) whose artifact writes
are legitimate, and a short, honest allowlist beats an aspirational
one. The scope is not static, though — the viewer top-level modules
(`mesh_viewer.py` has ~23 `plotter.render()` calls today,
`model_viewer.py` similar, plus `overlays/`) are exactly where V3/V4
bypasses would hide, so **each adoption slice widens the guard to
the modules it migrates**: guard scope grows in lockstep with
adoption, mirroring the allowlist burn-down.

### Part 6 — The mesh and model viewers join the same contract

The contract above is viewer-generic; the Dispatcher class is
already viewer-agnostic (pumps are injected). Adoption:

- Mesh and model viewers each construct their own Dispatcher with
  viewer-appropriate pumps and a small event vocabulary
  (entity-visibility, overlay-visibility, color-mode, filter
  events). `VisibilityManager.on_changed` and
  `OverlayVisibilityModel.on_changed` plain-callback lists migrate
  to dispatcher subscriptions; the models keep their state-owner
  role (Part 1) and gain owner-fired events (Part 2). Per Part 4,
  `VisibilityManager`'s mutators stop calling `_rebuild_actors()`
  inline — the rebuild becomes the mesh dispatcher's pump for the
  entity-visibility event, and the ~23 scattered `plotter.render()`
  calls in `mesh_viewer.py` collapse onto the dispatcher's coalesced
  render.
- The partially-deployed `core/_active_objects.py` Qt-signal hub
  (Plan 04) is **folded into the dispatcher's UI lane or deleted** —
  one cross-widget propagation mechanism per viewer, not two. Plan
  04's goal (decouple panel subscriptions from direct wiring) is
  exactly what dispatcher UI-lane subscriptions provide.
- Scales are named intent state with owners and mutators, not ad-hoc
  attributes: per-diagram runtime scale already conforms (diagram
  owns it; `DIAGRAM_MODIFIED` propagates); geometry deform scale
  already conforms (`GEOMETRY_DEFORM_CHANGED`); the mesh viewer's
  module-private `_overlay_scales` dict moves behind an
  overlay-style owner with a mutator that fires. **No global scale
  registry** — ownership stays local; it is the write path that
  unifies (see Rejected C).

## Invariants

- **INV-1** Every view-state field has exactly one owning object;
  widgets and write-only attributes are never the sole holder, and
  every panel can rebuild from owners alone.
- **INV-2** Intent and effective visibility are distinct channels:
  gate pushes go through `apply_effective_visibility` (or the
  equivalent artifact path) and never overwrite intent flags.
- **INV-3** Owner mutators fire their dispatcher event themselves;
  call sites never compensate with manual fires or raw renders, and
  any gesture mutating more than one owner field wraps in a
  dispatcher batch (`gesture_batch` for interactive cascades,
  `session_batch` for restore-scale bulk) so cost stays one pump +
  one render per gesture.
- **INV-4** `director.dispatcher` (and each viewer's dispatcher) is
  never `None`; no `getattr(..., "dispatcher", None)` defenses, no
  fallback render branches.
- **INV-5** In `viewers/ui/**`: no `render()`, no artifact-flag
  calls, no backend/pyvista/vtk imports — guard-tested with
  enumerated transitional allowlists.
- **INV-6** Degraded view fallbacks are loud: any path that
  knowingly renders something other than the requested state logs a
  warning (`scene.node_centroid_pass_failed` /
  `visibility.node_cloud_no_ownership_data` are the precedent).
- **INV-7** The reconciler is diff-shaped: in-place mutation fast
  paths are preserved and the existing dispatcher/in-place perf
  gates keep passing unchanged.

## Adoption (sequenced slices, each shippable)

- **V0 — shipped (PR #593, pre-ADR).** Gate revived through
  `apply_effective_visibility`; outline eye-toggles fire
  `LAYER_VISIBILITY_CHANGED`; deformed-ghost runtime state recorded
  and honored; silent node-cloud fallbacks made loud.
- **V1 — dispatcher-always + owners-fire (results viewer).**
  Optional-pump Dispatcher constructor + `bind(...)`;
  Director-constructed dispatcher; `DiagramRegistry.set_visible`
  fires; `gesture_batch()` added and the outline's composition /
  geometry eye cascades wrapped in it (perf gate: a composition
  toggle over N layers stays one pump + one render); session restore
  keeps `session_batch()`. Delete the seven `getattr` defenses, both
  raw-render fallbacks, and the now-redundant call-site fires in the
  settings tab and outline.
- **V2 — guard test.** `test_viewer_state_contract.py` (G-RENDER /
  G-ARTIFACT / G-IMPORT) over `ui/`, allowlist seeded from the
  measured baseline (five `viewer_window.py` renders + one pyvista
  import); burn-down list attached to the test.
- **V3 — mesh viewer.** Dispatcher + event vocabulary;
  `OverlayVisibilityModel` / `VisibilityManager` owner-fired events
  with `_rebuild_actors()` re-homed as the pump (Part 4);
  `_overlay_scales` behind an owner; guard scope widened to
  `mesh_viewer.py` + `overlays/`; allowlist burn-down.
- **V4 — model viewer + ActiveObjects disposition.** Same migration;
  fold `_active_objects.py` into the UI lane or delete it.
- **V5 — projection audit.** Sweep panels for widget-held state
  (checkbox restore paths, staged widget values) against INV-1;
  fix or document each.

V1+V2 are the load-bearing slices; V3–V5 ride the same rails.

## Rejected alternatives

- **A — A single global store with a serialized state tree
  (Redux-style).** One store object owning all view state across all
  three viewers, mutations as reducer actions. Rejected: the three
  viewers sit on genuinely different substrates (BREP entities /
  mesh cells / diagram layers) and a unified tree would re-fight the
  visibility-class unification this project already declined for
  good reasons; the dispatcher event matrix *is* the unidirectional
  loop, and what failed in practice was never the mechanism's shape
  but its optionality. The cheap, sufficient fix is making the
  existing machine mandatory.
- **B — Qt signals as the universal event substrate** (make owners
  QObjects). Rejected: the domain core must stay Qt-free — that is
  the headless-testability win ADR 0042 paid for, and the
  dispatcher's two-lane design already provides Qt-deferred
  coalescing where (and only where) the UI needs it. QObject
  inheritance would also leak Qt into modules the web target reuses.
- **C — A unified scale/visibility registry** ("one manager to rule
  them all"). Rejected: ownership locality is a feature — a
  diagram's scale belongs to the diagram, an overlay's scale to the
  overlay model. Central registries decouple state from the code
  that understands it and grow stringly-typed keys. The contract
  unifies the *write path and propagation*, not the storage.
- **D — Prose guidelines without guards.** Rejected on direct
  evidence: the dispatcher docstring already stated this ADR's core
  rule, and the outline tree violated it anyway. In a multiagent
  codebase, a rule without a failing test is a suggestion.

## Open questions

1. **Shared vs per-viewer event vocabularies (V3/V4).** Whether the
   mesh/model dispatchers reuse `_dispatch.py` constants where the
   semantics match (e.g. `ELEMENT_VISIBILITY_CHANGED`) or get
   namespaced vocabularies. Leaning: share the module, add kinds —
   the matrix is data, not behaviour. Decide at V3 with the code in
   hand.
2. **Guard scope growth beyond the adoption path.** V3/V4 widening
   to the migrated viewer modules is decided above; whether
   G-RENDER/G-ARTIFACT *additionally* widen to `diagrams/` call-site
   checks (diagrams may touch artifacts only via their own methods)
   is deferred until the V2 allowlist is burned down — widen only
   after the narrow guard is clean.
3. **Selection transport (V3/V4).** `SelectionState.on_changed` is a
   plain observer list separate from the dispatcher today. Whether
   selection-changed propagation moves onto the dispatcher UI lane
   or stays a dedicated channel (it has its own undo log and
   gesture grouping per ADR 0045) — decide per viewer during
   migration; ownership is settled by Part 1 either way.

   **Resolved at V4 (2026-06-10):** selection/focus propagation
   stays on its dedicated channel — `ActiveObjects` is **kept** as
   the per-viewer focus-state owner (active layer / geometry /
   composition / stage / step / pick mode / selection snapshot),
   not folded into the dispatcher UI lane and not deleted. The
   V4 census showed it is a conforming citizen of this contract,
   not a competitor: it *owns* focus state (Part 1), its setters
   fire its signals owner-side (Part 2), and it never touches
   render artifacts or `render()` (Part 4) — so the Part 6
   "folded … or deleted" clause is superseded by this resolution.
   The two mechanisms carry different state classes: the
   dispatcher carries view-state mutations to the reconciler;
   ActiveObjects carries focus changes to UI projections. One
   mechanism per *concern*, not per viewer.

## Consequences

- The bug class that motivated this ADR (state drift between UI
  surfaces, leaks past the gate, lost runtime toggles) becomes
  structurally unrepresentable rather than individually patched.
- Headless tests assert on events and owner state — the same code
  path the live viewer runs — instead of monkeypatched fallbacks.
- New panels/diagrams written by future agents inherit the contract
  from the guard test, not from reading this document.
- Cost: dispatcher fires move into owner mutators (small,
  mechanical), and the transitional allowlist must actually be
  burned down — an allowlist that only grows is the failure mode to
  watch in review.
