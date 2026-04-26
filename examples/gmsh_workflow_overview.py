"""
Gmsh Python API — workflow overview
====================================

A minimal end-to-end script that hits every step of the gmsh workflow:

    1. Initialize
    2. Create geometry (OCC recommended)
    3. Boolean operations
    4. Define conformal mesh (fragment)
    5. Mesh
    6. Extract data
    7. End

Example: a reinforced-concrete cross-section.  A box (concrete) and a
cylinder (rebar) are fragmented so they share their interface — that is
how the mesh inherits node-coincidence on the rebar boundary.
"""

import gmsh


gmsh.initialize()                                         # 1. INIT
gmsh.model.add("rc_section")

# 2. GEOMETRY  (OCC)
b = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
r = gmsh.model.occ.addCylinder(0.5, 0.5, 0, 0, 0, 1, 0.1)

# 3 + 4. BOOLEAN  +  CONFORMAL  (fragment shares interfaces)
out, _ = gmsh.model.occ.fragment([(3, b)], [(3, r)])
gmsh.model.occ.synchronize()

# physical groups (covered on PG slide)
vols = [t for d, t in out if d == 3]
gmsh.model.addPhysicalGroup(3, [vols[0]], name="concrete")
gmsh.model.addPhysicalGroup(3, [vols[1]], name="rebar")

# 5. MESH
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.15)
gmsh.model.mesh.generate(3)

# 6. EXTRACT DATA
nodes, coords, _    = gmsh.model.mesh.getNodes()
etypes, etags, conn = gmsh.model.mesh.getElements()
print(f"{len(nodes)} nodes, {sum(len(t) for t in etags)} elements")

gmsh.finalize()                                           # 7. END
