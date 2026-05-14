# 04 — ActiveObjects Singleton

**Status:** 🟡 **Steps 1 + 2 landed** via [PR #168](https://github.com/nmorabowen/apeGmsh/pull/168) (2026-05-14).
Step 1: `ActiveObjects` standalone with 7 typed signals. Step 2: outline composition + geometry selection
routed through it in `results.viewer`. Steps 3/4 (mesh/model migration) + more results-side signals
(`activeLayerChanged` for the settings tab, `activeStageChanged`, `activeStepChanged`) still pending —
see *Pending sub-tasks* below.
**Cost:** ~4 days  ·  **Depends on:** none  ·  **Architectural prereq for: 05, 06, 07**

## Goal

Introduce a per-viewer `ActiveObjects` singleton that holds the currently-active
selection, layer/diagram, view, and representation, and emits Qt signals when any change.
Every panel listens to those signals; no panel directly wires another.

## Why

Today, panels are wired together by explicit calls inside each viewer's constructor:

```python
# model_viewer.py (today, paraphrased)
sel.on_changed = self._on_sel_changed
self._on_sel_changed = lambda: (color_mgr.recolor_all(), browser_tab.refresh(), ...)
# ↑ A monolithic callback that knows about every dependent panel.
```

This pattern (the "closure capture" tension flagged in the architectural synthesis) means:

- Adding a new panel that cares about selection requires editing the central callback.
- `_rebuild_scene()` has to mutate managers in-place so closures keep working
  (`model_viewer.py:731-794`).
- The three viewers each invented their own selection-callback graph — none compatible.

ParaView pattern: one global `pqActiveObjects` singleton holds active source / view /
representation / port / selection and emits a Qt signal per change. Properties Panel,
Pipeline Browser, Color Editor, Info Panel — every panel subscribes independently. New
panel? Just connect to the signal; no central edit.

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "Click a source → Properties Panel updates"](../paraview-flows/index.html)
- Files:
  - `Qt/Components/pqActiveObjects.h:34` — the singleton class.
  - `pqActiveObjects.h:48-91` — accessors (activeView, activeSource, etc.).
  - `pqActiveObjects.h:129-159` — setters + signals.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/core/_active_objects.py` — the singleton (`QObject` subclass).

**Modify:**
- `src/apeGmsh/viewers/model_viewer.py` — replace the ad-hoc callback wiring with
  `active.selectionChanged.connect(...)` etc.; route picks through `active.set_selection(...)`.
- `src/apeGmsh/viewers/mesh_viewer.py` — same.
- `src/apeGmsh/viewers/results_viewer.py` — same; add `activeLayerChanged`,
  `activeGeometryChanged` for the post-solve outline.
- `src/apeGmsh/viewers/core/selection.py` — `SelectionState` emits via `ActiveObjects`
  instead of its own `on_changed` callback. Keep `on_changed` for one release as a
  compatibility shim; mark deprecated.

## API sketch

```python
# _active_objects.py
class ActiveObjects(QObject):
    """One per viewer window. Single source of truth for 'what is selected / active'."""

    selectionChanged       = Signal(object)   # SelectionState
    activeLayerChanged     = Signal(object)   # Layer | None    (results.viewer only)
    activeGeometryChanged  = Signal(object)   # Geometry | None (results.viewer only)
    activeStageChanged     = Signal(object)   # Stage | None    (results.viewer only)
    activeViewChanged      = Signal(object)   # ViewFrame
    # …expand as needed; one signal per kind, payload typed.

    @property
    def selection(self) -> SelectionState | None: ...
    def set_selection(self, sel: SelectionState | None) -> None: ...
    # …mirror for layer / geometry / stage / view
```

Not a process-global singleton. One instance per `ViewerWindow`, owned by it. Panels get
the instance via `viewer_window.active`. This keeps tests parallelizable.

## Implementation plan

1. Land `_active_objects.py` with the full signal surface and dummy data — nothing wired
   yet. Tests verify signals fire when setters are called.
2. Migrate `model_viewer.py` to use it. Verify all selection-driven UI still works.
   This is the riskiest viewer — most existing callback graph.
3. Migrate `mesh_viewer.py`. Add `activeElementSetChanged` if needed; consolidate the
   three pick-modes (brep/element/node) into one `selection` payload.
4. Migrate `results_viewer.py`. Add `activeLayerChanged`, `activeGeometryChanged`.
5. Mark `SelectionState.on_changed` deprecated. Don't remove yet.

## Risks

- **Big refactor.** Migrating three viewers touches a lot of files. Mitigation: land in
  four steps (above), each independently testable. Don't open a PR that does all three
  viewers at once.
- **Test coverage gap.** We don't have great coverage of the selection-callback graph
  today. Mitigation: before refactoring each viewer, write 2–3 integration tests that
  exercise the current behavior (pick an entity → check side-panel state). Then refactor.
  These tests stay valuable after the migration.
- **Memory leaks via Qt signals.** Forgetting to disconnect signals when a viewer closes
  leaks references. Mitigation: `ActiveObjects` is owned by `ViewerWindow`; both go away
  together. Use `connect(receiver, ...)` with explicit receiver so Qt auto-disconnects.
- **`SelectionState` already emits via a non-Qt callback (`on_changed`).** Two emission
  paths during the migration. Acceptable — call sites move one at a time; one release
  later, remove `on_changed`.

## Done criteria

- [ ] `ActiveObjects` exists with full signal surface, unit tests.
- [ ] All three viewers instantiate one `ActiveObjects` per window.
- [ ] All inter-panel notifications in `model_viewer.py`, `mesh_viewer.py`,
      `results_viewer.py` route through `ActiveObjects` signals. Grep verifies: no panel
      calls another panel's `refresh()` / `update()` directly.
- [ ] `_rebuild_scene()` in `model_viewer.py` no longer needs to mutate managers in-place
      to preserve closures — closures don't capture managers anymore.
- [ ] Adding a new test panel (e.g. a debug info panel) takes <20 LOC and zero edits
      to existing panels.
- [ ] No regression in existing selection / visibility / picking tests.

## Out of scope

- Cross-window active objects (e.g. results.viewer and mesh.viewer sharing selection).
  We don't need this and it adds complexity.
- Persisting active state across viewer launches.
- Replacing the existing undo/redo system. `SelectionState`'s undo stack stays as-is.

## Pending sub-tasks after PR #168

- **Step 2 cont.** — route the remaining results.viewer callback sites through
  `ActiveObjects`:
  - `active_layer` — when user picks an individual layer; drives `set_selected` on
    the settings tab.
  - `active_stage` — Director's stage observer cascade.
  - `active_step` — time scrubber → Director → diagram dispatch.
- **Step 3** — migrate `mesh.viewer`. Consolidate the three pick modes
  (brep/element/node) under a single `selection` payload + the
  `activePickModeChanged` signal already defined in step 1.
- **Step 4** — migrate `model.viewer`. Riskiest viewer: most complex existing
  callback graph (`_rebuild_scene` closure-capture pattern; multi-panel selection
  cascade). Per the original plan: migrate last so the API has been proven on
  results + mesh first.
