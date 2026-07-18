# 03 — Outline Tree Eye-Icon Visibility

**Status:** pending  ·  **Cost:** ~3 days  ·  **Depends on:** none

## Goal

Add a clickable eye-icon column to the outline trees in all three viewers, replacing
today's side-panel checkboxes / context-menu visibility toggles. One click in the tree
itself = visibility on/off.

## Why

- `model_viewer`: Browser tab lists physical groups with visibility toggled via a
  separate sub-panel.
- `mesh_viewer`: Mesh browser tab uses a similar split.
- `results_viewer`: Outline tree (Geometries → Compositions → Layers) hides visibility
  in the right-side Session panel — users hunt for it.

Three different UIs for the same operation. ParaView solves this with one clickable icon
in column 0 of the pipeline browser. Direct manipulation, discoverable, consistent.

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "Toggle eye icon on pipeline node → actor hides"](../paraview-flows/index.html)
- Files:
  - `Qt/Components/pqPipelineBrowserWidget.h` — flat tree view with the icon column.
  - `Qt/Components/pqPipelineModel.cxx` — model handles icon-column click.
  - `Qt/Core/pqDisplayPolicy.h` — visibility policy.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/ui/_outline_tree_icon_delegate.py` — a `QStyledItemDelegate` that
  paints the eye icon in column 0 and intercepts clicks on that column.

**Modify:**
- `src/apeGmsh/viewers/ui/_outline_tree.py` (`results.viewer`) — adopt the new delegate.
  Update the model to expose visibility state as `Qt.CheckStateRole` on column 0; remove
  the visibility checkboxes from `_session_panel.py`.
- `src/apeGmsh/viewers/ui/_browser_tab.py` (`model.viewer`, if that's where the physical-
  group browser lives) — same treatment. Same delegate.
- `src/apeGmsh/viewers/ui/_mesh_browser.py` (`mesh.viewer`) — same.

## API sketch

```python
# _outline_tree_icon_delegate.py
class EyeIconDelegate(QStyledItemDelegate):
    """Paints an eye / eye-slash icon in column 0; emits clicked(QModelIndex)."""

    icon_clicked = Signal(QModelIndex)

    def paint(self, painter, option, index): ...
    def editorEvent(self, event, model, option, index) -> bool: ...
    # ↑ returns True if the click hit the icon and we handled it
```

Model contract: each row implements
```python
def is_visible(self, index: QModelIndex) -> bool: ...
def set_visible(self, index: QModelIndex, visible: bool) -> None: ...
```

When `set_visible` is called, the model emits `dataChanged` for column 0 (so the delegate
repaints) and the underlying scene actor's visibility is flipped.

## Risks

- **Click-target size.** Icons in tree cells are small. Make the hit area generous
  (the whole icon-column cell, not just the icon glyph). Add hover state for discoverability.
- **Hierarchical visibility.** When a Composition is hidden, do its child Layers go
  invisible? ParaView's answer: yes, parent visibility wins. Implement the same — but
  preserve each child's own visibility state, so toggling the parent back on restores
  the children's individual state. Document the rule in the user-facing tooltip.
- **Inconsistent backend.** `results.viewer` actors live behind the `DiagramRegistry`;
  `model.viewer` actors live behind the `EntityRegistry`. The delegate is generic — only
  the model knows how to flip the actor. Each viewer's model implements `set_visible`
  appropriately.
- **Theme.** Eye-open / eye-closed glyphs from `qtawesome` or hand-rolled SVG; respect
  light/dark theme. If `qtawesome` is not a dep, ship two small SVG assets.

## Done criteria

- [ ] Outline tree in `results.viewer` shows an eye icon in column 0 for every row
      (Geometry, Composition, Layer).
- [ ] Clicking the icon toggles the corresponding actor's visibility (verifiable by
      screenshot or a programmatic check on `actor.visibility`).
- [ ] Hiding a Composition hides all its child Layers visually but does not erase the
      Layers' own `visible` flags; un-hiding the Composition restores the prior state.
- [ ] Same behavior in `model.viewer` Browser (physical groups) and `mesh.viewer` Browser
      (BRep groups + element-type filters).
- [ ] Old visibility checkboxes / context-menu items removed from side panels.
- [ ] Eye icon respects dark/light theme.
- [ ] Tooltip on the icon explains the hierarchical rule.

## Out of scope

- Drag-to-reorder of outline rows (different feature, not blocked by this).
- Multi-select visibility toggle (Ctrl+click on icon to toggle a range). Could be a
  follow-up if useful.
- Visibility undo/redo. Today no viewer has undo for visibility; keep that consistent.
- Right-click context menu redesign — leave as-is.
