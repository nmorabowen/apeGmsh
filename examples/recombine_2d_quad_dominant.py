"""
Recombine alone — quad-dominant 2D mesh
=========================================

A flat rectangle, default 2D mesher, with `setRecombine` enabled on the
surface.  The Blossom recombination algorithm pairs adjacent triangles
into quads after the standard triangular mesh is generated.

Result: mostly quads, possibly with a few stray triangles where pairing
failed.  This is the typical "quad-dominant" outcome when you don't
constrain the topology with transfinite meshing.
"""

import gmsh

gmsh.initialize()
gmsh.model.add("plate_quad")

plate = gmsh.model.occ.addRectangle(0, 0, 0, 5, 3)
gmsh.model.occ.synchronize()

# turn triangles into quads on this surface
gmsh.model.mesh.setRecombine(2, plate)

gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
gmsh.model.mesh.generate(2)

gmsh.fltk.run()
gmsh.finalize()
