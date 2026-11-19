### ADDED — ADR 0138 (Proposed): apeGmshViewer selection. No code changes.

- Design only: one selection per target kind (node, element, Gauss point, hinge, fibre), one active kind, and a current
  target that single-target tools follow.
- Picking reads an ID image drawn on a press, and regions read positions projected at the press. Visibility stays per element,
  and hide and isolate act on the elements that own a target.
- Adds the mouse and region rules, an indexed highlight, what a tool sees of the selection, and the slice order.
- `internal_docs/spike_viewer_selection.md` holds the measurements behind it.
