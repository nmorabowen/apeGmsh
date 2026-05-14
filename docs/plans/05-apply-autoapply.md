# 05 — Apply / Reset / Auto-Apply on Diagram Settings

**Status:** ✅ **Landed** via [PR #168](https://github.com/nmorabowen/apeGmsh/pull/168) (2026-05-14).
Investigation revealed the Apply transaction was already implemented (the `_pending_appliers` list +
per-card Apply button); the work narrowed to adding Reset, Auto-Apply toggle + persistence, debounce
infrastructure, and migrating 4 representative widgets to the new `_stage_with_signal` helper. Other
widgets still use the bare `_pending_appliers.append` idiom — both paths coexist and Apply works
identically either way.
**Cost:** ~3 days estimated → ~1 day actual (scope shrank after discovery).
**Depends on:** none (no longer requires plan 04 — discovered the existing Apply pattern works
standalone).

## Goal

Replace the current "every property change applies immediately" pattern in the diagrams
settings panel with an explicit Apply / Reset transaction, plus an Auto-Apply toggle for
users who prefer live preview.

## Why

Today, every checkbox or scale-spinner change in the Diagrams tab triggers an immediate
`update_to_step` on the affected Diagram. For cheap diagrams (contour, deformed shape)
this is fine. For expensive ones (fiber-section panels, section cuts that read OpenSees
element-by-element) it stutters the UI. Worse, if the user is twiddling several knobs in
sequence, each one runs a full update.

ParaView pattern: properties accumulate as "unchecked" values; Apply commits them all in
one transaction; Auto-Apply (toggle) reverts to live preview when the user wants it.

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "Edit property → Apply → pipeline updates → view re-renders"](../paraview-flows/index.html)
- Files:
  - `Qt/ApplicationComponents/pqApplyBehavior.h` — the apply transaction.
  - `Qt/Components/pqPropertiesPanel.h:32` — the panel hosting Apply/Reset buttons.
  - `vtkSMProperty.h:22-34` — checked vs. unchecked value semantics.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/ui/_apply_bar.py` — the Apply / Reset / Auto-Apply toggle widget
  that mounts at the top of any settings panel.
- `src/apeGmsh/viewers/diagrams/_pending_changes.py` — buffer that holds pending property
  edits for a Diagram until Apply or Auto-Apply commits them.

**Modify:**
- `src/apeGmsh/viewers/ui/_diagram_settings_tab.py` — wrap per-diagram settings in the
  Apply bar; route property edits through `PendingChanges` instead of immediately calling
  `diagram.update_to_step`.
- `src/apeGmsh/viewers/diagrams/_base.py` — add `apply_pending(changes: dict)` method
  to `Diagram` that takes a batch and triggers one update; deprecate per-property
  immediate setters.

## API sketch

```python
# _pending_changes.py
class PendingChanges:
    """Buffers edits to a Diagram's style/selector until commit."""

    def __init__(self, diagram: Diagram): ...
    def stage(self, key: str, value: Any) -> None: ...
    def is_dirty(self) -> bool: ...
    def reset(self) -> None: ...
    def commit(self) -> None:
        """Apply buffered changes to the diagram in one update_to_step pass."""

# _apply_bar.py
class ApplyBar(QWidget):
    """[ Apply ] [ Reset ] [x] Auto-Apply"""

    def __init__(self, pending: PendingChanges, settings: QSettings): ...
    # Auto-Apply state persists per-user via QSettings — viewer-wide, not per-diagram.
```

Auto-Apply, when on, calls `pending.commit()` on every `pending.stage()` (with a small
debounce, ~150ms, to coalesce rapid edits like dragging a slider).

## Risks

- **Discoverability.** Users who don't notice Auto-Apply is off will be confused why
  their edits "do nothing." Mitigation: Apply button shows a pulsing dot indicator when
  changes are pending; tooltip says "N unapplied changes." On first launch, Auto-Apply
  defaults to ON (preserves current behavior); experienced users turn it off.
- **Existing diagrams update on every property assignment via observers.** Need to audit
  `_base.py`, `_contour.py`, etc. for places that bypass the settings tab and mutate
  diagram state directly. Those paths must either route through `PendingChanges` or
  document why they don't (e.g. time-scrubber-driven updates are not user edits).
- **Reset semantics.** Reset = revert pending edits to last applied state, not to
  defaults. Restore-Defaults is a separate button (ParaView has both); we ship only Reset
  in this plan, defer Restore-Defaults.

## Done criteria

- [ ] Diagram settings panel shows Apply / Reset / Auto-Apply toggle at the top.
- [ ] With Auto-Apply OFF: edits buffer; Apply button enables; pulsing dot visible;
      clicking Apply commits all edits in one `update_to_step`.
- [ ] With Auto-Apply ON: edits apply live with ~150ms debounce on rapid changes
      (slider drag should result in a small number of updates, not one per pixel).
- [ ] Reset reverts buffered edits without touching the diagram.
- [ ] Auto-Apply preference persists across viewer launches via `QSettings`.
- [ ] Performance: dragging the time scrubber across 100 steps with 3 diagrams on
      remains fluid (no per-property re-renders piling up).
- [ ] No regression in existing diagram tests.

## Out of scope

- Restore-Defaults button (would need a "defaults" snapshot per diagram type — defer).
- Property-level undo/redo. The whole `PendingChanges` buffer is one undo unit at most;
  fine-grained per-property undo is a follow-up.
- Apply behavior on the Stage / Time scrubber. Those are not "settings edits" and stay
  live.
- Apply on Geometry deformation toggles. Those are cheap and stay live (live preview is
  the whole point of the deform UI).

## Deferred from this implementation (PR #168)

- **Remaining widget migrations.** 4 of ~18 widget call sites now use
  `_stage_with_signal` (Contour: cmap/clim-min/clim-max/opacity; Deformed: scale).
  The other panels keep the bare `_pending_appliers.append` idiom — Auto-Apply
  is a no-op for them until they're migrated. ~2 hours of mechanical work whenever
  someone wants a particular widget to live-preview.
