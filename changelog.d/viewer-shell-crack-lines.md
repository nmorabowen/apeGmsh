### ADDED — apeGmshViewer traces continuous crack lines on layered shells (crack plots, slice 3)

The Cracks card's shells now default to continuous crack lines (`style: "lines"`), drawn the way the maintainer's reference figures are (CRACK_PLOTS_SPEC.md §3.3). The shells are grouped into planar panels: a wall, a slab level. In each panel and on each face, the outer concrete layer's widest crack (C1) gives a trace direction and an opening at every Gauss point.
- **Tracer:** evenly spaced streamlines (Jobard and Lefer). Each Gauss point's crack axis is sign-free and is flipped to agree with the travel direction; the field is the opening-weighted mean of the cracked neighbours. A line stops at an uncracked or empty zone, at a turn sharper than 40°, near another line, or at the mesh's edge. Seeds are taken widest first.
- **Scale:** every length is in units of the Gauss pitch, so a model in mm and one in m look alike. The spacing is a drawing choice; the direction and the cracked extent are the model's.
- **Placement:** each line vertex is located in its element as natural coordinates, so the lines follow the deformed shape through the element's own shape functions and are not re-traced when the deformation scale changes.
- **Drawing:** lines are ribbons lying on their face, at ±t/2 like the S2 ticks, red or blue by face. They are wider where the crack opens more, on a log scale. `style: "ticks"` keeps the S2 tick drawing.
- **Tests:** the tracer is checked against the spec's four cases (a uniform field, a radial field, sign flips, a hole).
