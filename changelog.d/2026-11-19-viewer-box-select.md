### ADDED — apeGmshViewer: box selection, hover and a selection bar (ADR 0138 SEL-1b)

- A left drag draws a selection box. Left to right takes what is fully inside it; right to left takes what it touches.
  It takes what is visible: hidden elements, elements inactive at the stage shown, and anything a section plane cuts away
  are never taken. The count shows while dragging, and Esc drops the box.
- Shift adds and Ctrl removes, for a click and for a box. A Shift+click no longer takes a selected element out.
- The element or node under the cursor is marked and named before any click.
- A selection bar above the Windows menu shows which kind clicks and boxes act on, how many of each are selected, and
  clears the active kind. The 1 and 2 keys choose elements or nodes.
- While Nodes is active, the element selection is drawn faintly, so the node markers read over it.
- A box reads an ID image drawn once at the press. Clicks still use the ray cast (ADR 0138 D5).
