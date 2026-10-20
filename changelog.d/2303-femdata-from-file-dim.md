### FIXED — FEMData read from a results file gives each element group its parent dimension from the response catalog: a LadrunoLST group is a surface, not a wedge (#2303)

`Results.from_ladruno` and `Results.from_mpco` build a FEMData from the
file's own `MODEL/` group. Each group's `dim` used to be a class-name guess.
That guess made a `LadrunoLST` group (whose `TOPOLOGY` is `"custom"`) a 3-D
cell, and a `LadrunoCST` group in a 3-D model a line. The effect was a
silent wrong answer, not only a drawing one: `fem.elements.select(dim=2)`
came back empty, `select(dim=3)` returned the triangles, and the static
facet plot dropped the group.

A catalogued class now takes its catalog parent dimension: the width of its
gauss-level `natural_coords`, 1 for the line-station, element-force, fiber
and zero-length families, and 2 for layered shells. An uncatalogued class
takes the `.ladruno` `TOPOLOGY` it declares. When a class has neither, its
dim is still the name guess, but an `ElementDimUnknownWarning` names it.
A declared `TOPOLOGY` that contradicts the catalog is also warned, and the
catalog's dim is used.
