### ADDED — apeGmshViewer: element contours carried to the nodes (V4c2b, #1875)

The three element-field draw modes V4c2a listed as "not yet available" now
draw, as the Qt `ContourDiagram` does, ported from
`results/_gauss_extrapolation.py`. **At nodes** (`gauss_node`) extrapolates
each element's Gauss values to its corners through the pseudo-inverse of the
linear corner shape functions at the group's `_natural_coords` (higher-order
types project onto their linear corners, as in Python) and averages them at
the nodes across elements. **At nodes, per element** (`gauss_node_discrete`)
keeps each element's own corner values, so jumps between elements show.
**Cell, averaged at nodes** (`gauss_cell_averaged`) averages one-point cell
values at the nodes; on a field with several Gauss points it averages each
element's mean (`element_mean_averaged`, an app presentation with no Qt
equivalent). The step range and the whole-history scan measure the values
drawn, not the Gauss values. An element with no row, or with a non-finite
Gauss value, is drawn as no data and adds nothing to its nodes' averages.

An element type with no corner shape functions (a pyramid, a line3), or
natural coordinates with too few columns, reads **"unsupported"** in those
modes, never Python's element-mean fallback. Only verified files draw, as
before. Python is the oracle: `fixtures/extrap.oracle.json`, written by
`scripts/make_viewer_results_fixtures.py --extrapolation`, holds apeGmsh's
own extrapolation of the committed `extrap_*` fixtures (hex8 at 8 points,
quad4 at 4, tri3 and tet4 at 1), and the app equals it to 1e-9 relative.
