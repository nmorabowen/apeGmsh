### ADDED — apeGmshViewer: element (Gauss) contours, flat per cell (V4c2a, #1833)

In contour mode the legend's field list now offers the element fields of the
results stage shown, from its `gauss_points` groups, beside the nodal ones.
Component names come from the file (`[k]` for an unnamed one, "(local)" when
the name says its axes are local). An element field is drawn flat per cell:
`gauss_cell`, the element's one Gauss value when its group has one point, or
`gauss_cell_averaged`, the mean of the element's Gauss values; the legend
names the topology and the averaging. Each `_element_index` row is joined to
the drawn cell by its FEM element id, never by position, and one field
across several element types (quads and triangles) colours each group's
cells from its own group. A drawn element no group records, or one with a
non-finite Gauss value, is drawn as no data. The colour range modes, the
whole-history scan (of the selected element field only), the deformed shape,
stage activation and re-reads work as for nodal contours.

Element contours draw **only** for a results file whose root
`element_index_space` is the stamp `"fem"`. An unverified file is refused:
the banner says the file does not prove which elements its rows are, the
mesh is drawn as no data, and main never sends its element rows or values. A
field is also refused, with a typed state and the reason, when its groups
disagree on its components (a component missing, or refused as a partial
write, in one group), when an `_element_index` id is not a cell of the
model, or when an id has two rows. The extrapolations to the nodes
(`gauss_node`, `gauss_node_discrete`) are listed and say "not yet available"
(V4c2b). `apeGmshViewer/MEASUREMENTS.md` has the element scrub rows (worst
185 steps/s deformed, peak 360 MB), and
`scripts/make_viewer_results_fixtures.py --elements` writes the fixtures.
