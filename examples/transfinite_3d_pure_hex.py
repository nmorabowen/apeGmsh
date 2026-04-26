"""
Transfinite + recombine 3D — pure hex mesh
=============================================

A box meshed as pure hexahedra by marking every entity transfinite +
recombine.  This is the canonical pattern for 3D hex meshing in gmsh:
the geometry must be hex-meshable (box, swept solid, mapped block) and
every dimension — curves, surfaces, volume — must be configured.

For arbitrary CAD geometry that doesn't fit a hex-meshable topology,
this pipeline cannot produce hexes; you fall back to tetrahedra (or to
a hex-dominant mesh with embedded tets, which is its own subject).
"""

import gmsh

gmsh.initialize()
gmsh.model.add("box_hex")

box = gmsh.model.occ.addBox(0, 0, 0, 5, 3, 2)
gmsh.model.occ.synchronize()

# 6 nodes along every edge
for d, t in gmsh.model.getEntities(1):
    gmsh.model.mesh.setTransfiniteCurve(t, 6)

# every face: structured + quad recombine
for d, t in gmsh.model.getEntities(2):
    gmsh.model.mesh.setTransfiniteSurface(t)
    gmsh.model.mesh.setRecombine(2, t)

# the volume: structured + hex recombine
gmsh.model.mesh.setTransfiniteVolume(box)
gmsh.model.mesh.setRecombine(3, box)

gmsh.model.mesh.generate(3)

gmsh.fltk.run()
gmsh.finalize()
