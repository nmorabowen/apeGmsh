### ADDED — `apeGmsh.interop.stko`: read STKO `.scd` documents

`read_scd(path)` reads an STKO CAE document (HDF5) into frozen dataclasses:
geometries with their per-sub-shape element / physical property and
local-axes assignments, the mesh (nodes, element records, which elements
each sub-shape generated, orientation quaternions), selection sets,
physical / element properties, conditions (with their geometry or
interaction assignments), interactions, local axes, definitions and
analysis steps. Parameters keep STKO's names and decode every encoding
the documents use (`BOOL` / `INT` / `REAL` / `QNT_SCALAR` / `INDEX`,
`STRING` / `QNT_VEC3` / `QNT_VECN` / `INDEX_VEC`, nested custom objects).
`ScdModel.set_elements` / `set_nodes` expand selection sets the way STKO
writes them to `.mpco.cdata`; `analysis_elements()` separates what OpenSees
receives from the rest of the mesh (shell edge meshes, solid face meshes).
`write_brep(scd, out, geometry=None)` extracts the OCC geometry as text
BREP for `g.model.io.load_brep`; it needs the new `apeGmsh[stko]` extra
(`cadquery-ocp`), kept out of `[all]`.

Checked on the San Ramon documents: 1A gives 13,704 analysis elements and
4D 57,906 (shells, fiber columns, soil bricks, 7,324 embedded links), both
matching STKO's own Tcl exports; STKO face / edge *i* is gmsh tag *i + 1*,
so the 192 column edges of 1A land on gmsh's 192 free curves.
