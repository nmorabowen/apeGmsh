"""
Transfinite + recombine — pure 2D structured quad mesh
========================================================

Same flat rectangle, but every bounding curve is marked transfinite
(fixed node count) and the surface is marked transfinite + recombine.
The result is a perfectly regular grid of quads — no stray triangles,
no irregular spacing.

Transfinite surfaces require 4-sided topology (3-sided is also allowed
but degenerates).  For surfaces with > 4 boundary curves you must pass
``cornerTags`` explicitly to identify the corners.
"""

import gmsh

gmsh.initialize()
gmsh.model.add("plate_struct")

plate = gmsh.model.occ.addRectangle(0, 0, 0, 5, 3)
gmsh.model.occ.synchronize()

# fix node count per edge — sets the grid resolution
for d, t in gmsh.model.getEntities(1):
    gmsh.model.mesh.setTransfiniteCurve(t, 11)

# structured layout + recombine to quads
gmsh.model.mesh.setTransfiniteSurface(plate)
gmsh.model.mesh.setRecombine(2, plate)

gmsh.model.mesh.generate(2)

gmsh.fltk.run()
gmsh.finalize()
