### ADDED — apeGmshViewer: box selection through the model (ADR 0138 SEL-2)

- A Through toggle on the selection bar makes a box select every drawn element or node in it, whether something in front
  hides it or not. Left to right takes the elements with every vertex inside; right to left, every element reaching into
  the box (a bounding test, so an element that only passes near a corner can be taken).
- Through or not, a box never takes a hidden element, an element inactive at the stage shown, or what a section plane
  cuts away. The box's count says "through" while the toggle is on. A click still takes what is drawn under the cursor.
