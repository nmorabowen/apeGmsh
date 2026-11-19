### FIXED — apeGmshViewer: a click takes the element drawn under the cursor (ADR 0138 SEL-2)

- A click, and the hover mark that names what it would take, read an ID image of what is drawn, as the box does. On a
  model with beams over a slab, a click no longer takes a beam hidden behind the slab, and on a beam split into short
  segments it takes the segment under the cursor, not the one beside it.
- A beam drawn within 4 pixels of the cursor still takes the click from the face under it, so beams stay as easy to hit.
