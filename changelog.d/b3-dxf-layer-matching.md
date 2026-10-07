### FIXED — DXF import: layer physical groups hold their curves again (#1532)

`g.model.io.load_dxf` matched each surviving curve to its DXF layer by a
bounding box rounded to 8 decimals, but OCC pads `gmsh.model.getBoundingBox`
by 1e-7, so every curve landed in `_unmatched` silently and the layer groups
came out empty. The match is now geometric: the curve type, exact endpoint
coordinates (`getBoundary`/`getValue`, which OCC does not pad) and the
curve's exact box for lines, arcs and circles (a B-spline is matched by its
endpoints inside its control-polygon hull), under a tolerance floored at ten
times the OCC pad and at `point_tolerance`, plus a round-off term of 1e-12
of the drawing's extent.
Arcs now record their true box (axis extremes included), not their chord's.
A DXF entity that no imported curve matches, or a curve that matches
entities on several layers (a duplicate drawn on two layers: it joins every
such group), raises a `WarnDxfLayerMismatch` warning that names the layers.
`tests/test_dxf_layer_matching.py` pins it; program slice B3-b (#1556).
