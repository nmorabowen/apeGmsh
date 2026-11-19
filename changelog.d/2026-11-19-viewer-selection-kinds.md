### CHANGED — apeGmshViewer: a node selection beside the element selection (ADR 0138 SEL-1a)

- The Sets panel's Nodes / Elements toggle now chooses what a click selects. With Nodes, a click selects the node nearest
  it, where it is drawn, so picking nodes works on the deformed shape too. Each kind keeps its own selection, and Esc clears
  the active one.
- A node set saves the selected nodes. It no longer guesses a node from where an element was clicked, so a selection made
  in Search no longer leaves Nodes empty, and the refusal after a click on the deformed shape is gone.
- Shift+click on an element selected through its group now takes it out of the selection. The group gives way to its other
  elements.
- Adding to a large selection is linear. Adding one element to 26,970 took about 2.8 s before.
- The highlight is an index over the drawn faces and draws each element's outline from node-tag pairs. Selecting every
  element of a 27,000-element model no longer builds one map of coordinate strings. Hidden elements are not highlighted.
