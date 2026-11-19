### ADDED — apeGmshViewer: selection undo (ADR 0138 SEL-2)

- Ctrl+Z undoes the last selection change, and Ctrl+Y or Ctrl+Shift+Z redoes it. A click on empty space, an Esc or a box
  that replaced a large selection can be taken back. The selection bar has an Undo button.
- Undo covers the element and the node selection in one history, and makes the kind it changes active, so the change shows.
  It keeps the last 100 changes; opening or re-reading a model starts it again. A text field keeps its own Ctrl+Z.
- The selection bar's hint is shorter ("Shift adds · Ctrl removes"), so the bar fits beside the right rail; the box rule
  (left to right inside, right to left touching) moved to its tooltip, and the box's count still says which it takes.
