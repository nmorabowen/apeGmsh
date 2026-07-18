# Future: Interactive Section Cut Editing

**Status:** future · **Estimated cost:** ~1 week · **Depends on:** plan 04
**Discussed in:** [`../region-tools-comparison.md`](../region-tools-comparison.md)

## Goal

Allow dragging the section-cut plane in 3D and re-running STKO integration on release,
producing updated F/M time histories at the new plane.

## Why

- Today: section cuts are defined via API (`SectionCutDef`) and visualized statically.
  Editing requires going back to code.
- The natural marriage of our engineering-cut feature with ParaView-style
  interactivity. Closes a gap that no other open-source FEM postprocessor fills.

## Mechanism

- Add a `vtkPlaneWidget` to `SectionCutDiagram` in "edit mode" (toggleable via panel).
- On widget **release** (not during drag — STKO integration is expensive):
  - Re-emit `SectionCutDef` with the new plane.
  - Call `.to_spec()` → STKO_to_python integration.
  - Update the cached `SectionCut(F, M, time, …)` and re-render the F/M plot pane.
- Display the new time history in a docked plot pane next to the cut quad.

## Files

- Modify: `src/apeGmsh/viewers/diagrams/_section_cut.py` — add edit mode + widget +
  live re-integration hook.
- Modify: `src/apeGmsh/viewers/diagrams/_session.py` — persist edited plane.
- New: integration layer with `apeGmsh.cuts` for live `.to_spec()` re-call.
- New (or extend `_plot_pane.py`): F/M time history plot for the cut.

## Risks

- **Integration cost.** For large models, STKO integration can take seconds. UI must
  show progress + disable widget controls during run.
- **Concurrent edits.** If user drags two cut planes rapidly or releases mid-run,
  throttle. Cancel-and-restart on next release.
- **Persistence roundtrip.** A session-saved edited cut needs to roundtrip cleanly
  through both `SectionCutDef` (geometry + element filter) and the live-edit state
  (current plane).

## Done criteria

- Section cut diagram has an "Edit" toggle; toggling enables plane-widget handles.
- Dragging + releasing updates the integration result.
- Docked plot pane shows F/M time histories at the active cut.
- Edited state persists across viewer launches.

## Out of scope

- Editing the bounding polygon (just the plane normal + origin).
- Multi-cut sweeps via the widget (sweeps remain programmatic).
- Live integration during drag (always release-driven).
- Editing the underlying element filter set.
