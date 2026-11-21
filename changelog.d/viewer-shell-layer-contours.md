### ADDED — apeGmshViewer contours layered-shell results by face or over the layers (crack plots, slice 4)

The Contour card can now colour layered shells by any of their layer results: damage (`d+`, `d-`), layer stress and strain, and crack openings. Each layer group is offered as three element fields:
- `<result>@+face`: the outer responding layer on the face toward the positive global axis the shell's normal is most along (a slab's top, a wall's +x or +y side), the red face of the crack plots;
- `<result>@-face`: the other outer layer;
- `<result>@max`: the largest value over every responding layer, for a quantity such as damage.

The reader reduces the layers, so each field draws, scans its range and smooths to nodes like any element field. The face fields need the shells' frames (`MODEL/LOCAL_AXES`, fork #1133). In a file without them those fields are refused by name, and an element with no frame reads as no data. Part sets stitch the fields from their parts, as they do layer groups.
