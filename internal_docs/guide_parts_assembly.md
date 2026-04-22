# Working with Parts to Build an Assembly

apeGmsh follows an Abaqus-inspired Part/Assembly philosophy: geometry is built in
isolated **Parts**, then imported and combined inside a **apeGmsh session** that
acts as the assembly.  This separation keeps geometry reusable, meshing
independent, and constraints declarative.

## Core Concepts

**Part** is a pure geometry builder.  It owns an isolated Gmsh session where you
create points, curves, surfaces, and volumes.  It has no meshing, no physical
groups, and no solver awareness.  When you are done, you export the geometry to a
STEP file.

**apeGmsh** (the main session) is the assembly.  It imports Parts, positions
them, fragments interfaces so the mesh is conformal, generates the mesh, defines
constraints, and exports to a solver.  Everything that happens after geometry is
done—meshing, physical groups, mesh selections, constraints, OpenSees export—lives
here.

**STEP** is the contract between the two.  STEP preserves the full parametric
OCC geometry (exact NURBS, topology, tolerances), so the assembly can re-mesh the
imported geometry with any settings, apply boolean operations, and set transfinite
or structured meshing after the fact.

## Phase 1 — Define Parts

Each Part runs in its own Gmsh session.  You build geometry through the familiar
`part.model` API, then save to disk.

```python
from apeGmsh import Part

# ── Column ────────────────────────────────────────────────
column = Part("column")
column.begin()

column.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0)

column.properties["material"] = "concrete"
column.properties["section"]  = "500x500"

column.save()          # → column.step
column.end()

# ── Beam ──────────────────────────────────────────────────
beam = Part("beam")
beam.begin()

beam.model.geometry.add_box(0, 0, 0,  6.0, 0.3, 0.5)

beam.properties["material"] = "concrete"
beam.properties["section"]  = "300x500"

beam.save()            # → beam.step
beam.end()
```

**Key points:**

- `Part.model` gives you the full OCC geometry API (add_point, add_line, add_box,
  add_cylinder, extrude, etc.).
- `Part.properties` is a free-form dict for metadata (material, section type,
  thickness, etc.) that carries through to the assembly.
- `Part.save()` defaults to `"{name}.step"`.  You can pass a custom path or force
  IGES format with `fmt="iges"`.
- Always call `part.end()` when done to release the Gmsh session.

## Phase 2 — Create the Assembly Session

Open a apeGmsh session and import parts via `g.parts`.  There are multiple entry
points depending on where your geometry comes from.

### Entry Point A — Import a Part object

The most common path.  The Part must be saved to disk first.

```python
from apeGmsh import apeGmsh

g = apeGmsh("frame")
g.begin()

# Place two columns
g.parts.add(column, label="col_left",  translate=(0, 0, 0))
g.parts.add(column, label="col_right", translate=(5.5, 0, 0))

# Place beam on top
g.parts.add(beam, label="beam_top", translate=(0, 0, 3.0))
```

The `add()` method imports the Part's STEP file into the current Gmsh session,
applies translation and rotation, and registers the resulting entities under a
label.  If you omit `label`, it auto-generates one as `"{part.name}_1"`,
`"{part.name}_2"`, etc.

### Entry Point B — Import a STEP file directly

When you have a CAD file from an external tool (FreeCAD, Rhino, SolidWorks):

```python
g.parts.import_step("slab.step", label="slab", translate=(0, 0, 3.5))
```

### Entry Point C — Inline geometry (context manager)

For parts that don't need reuse, build geometry directly in the assembly session
and wrap it in a tracking block:

```python
with g.parts.part("foundation"):
    g.model.geometry.add_box(-0.5, -0.5, -0.5,  7.0, 1.5, 0.5)
```

Everything created inside the `with` block is automatically recorded as a named
instance.

### Entry Point D — Adopt existing geometry

After loading a STEP through `g.model.io.load_step()`, retroactively tag the
imported entities:

```python
g.model.io.load_step("bracket.step")
g.parts.from_model("bracket")
```

You can also adopt specific entities by dimension and tag:

```python
g.parts.from_model("slab", dim=3, tags=[1, 2])
```

### Entry Point E — Manual registration

Tag entities you already have in hand:

```python
g.parts.register("rebar", [(1, 10), (1, 11), (1, 12)])
```

## Phase 3 — Positioning

Translation and rotation are applied at import time through keyword arguments on
`add()` and `import_step()`.

```python
import math

g.parts.add(column, label="col_1", translate=(0, 0, 0))
g.parts.add(column, label="col_2", translate=(6, 0, 0))
g.parts.add(column, label="col_3",
            translate=(3, 0, 0),
            rotate=(math.pi/4, 0, 0, 1))   # 45° about Z
```

Rotation format is `(angle_rad, ax, ay, az)` for rotation about the origin, or
`(angle_rad, ax, ay, az, cx, cy, cz)` to specify a custom center of rotation.
Rotation is applied before translation.

## Phase 4 — Fragment for Conformal Meshing

When two parts share an interface (a beam sitting on a column, a slab touching a
wall), their meshes must be conformal at that interface.  `fragment_all()` performs
a boolean fragmentation that splits entities at their intersections and produces a
single conformal topology.

```python
g.parts.fragment_all()
```

This automatically detects the highest dimension present and fragments at that
level.  Instance entity tags are updated in-place, so all subsequent operations
(physical groups, mesh generation, constraint resolution) see the post-fragment
tags.

For selective fragmentation between two specific parts:

```python
g.parts.fragment_pair("beam_top", "col_left")
```

**Important:** Any entities in the session that are not tracked by a part label
will trigger a warning during fragmentation.  Use `g.parts.from_model()` or
`g.parts.register()` to adopt orphan entities before fragmenting.

## Phase 5 — Physical Groups and Mesh Generation

### Automatic physical groups from parts

The simplest path—one physical group per instance label:

```python
g.parts.add_physical_groups()
# → {"col_left": 1, "col_right": 2, "beam_top": 3, "foundation": 4}
```

### Manual physical groups

For finer control (e.g., tagging surfaces for boundary conditions), use the
standard `g.physical` API:

```python
g.physical.add("base_fixed", dim=2, tags=[...])
g.physical.add("load_surface", dim=2, tags=[...])
```

### Mesh generation

```python
g.mesh.sizing.set_size_global(min_size=50, max_size=200)
g.mesh.generation.generate(dim=3)
```

At this point you can also apply transfinite meshing, recombine quads, set mesh
fields, or use any other Gmsh meshing feature through `g.mesh` or direct
`gmsh.model.mesh` calls.

## Phase 6 — Mesh Selections (Post-Mesh Named Sets)

After meshing, you may need named sets of nodes or elements that don't correspond
to geometry (e.g., "all nodes on plane Z=3.0" or "all elements inside a box").
`g.mesh_selection` provides spatial queries:

```python
# Nodes at the beam-column interface
g.mesh_selection.add_nodes(name="interface_nodes", on_plane=("z", 3.0))

# Elements inside a region
g.mesh_selection.add_elements(dim=3, name="core_elements",
                              in_box=(1, 0, 0, 5, 0.5, 3))

# Set algebra
g.mesh_selection.union(0, tag_a, tag_b, name="all_support_nodes")
```

Mesh selections use the same `(dim, tag) + name` identity as physical groups but
live in an independent namespace.

## Phase 7 — Constraints

Constraints follow a two-stage pipeline: **define** before or after meshing
(they are geometric intent), then **resolve** after meshing to get concrete node
pairs and weights.

### Define constraints

Constraint labels can reference part labels, physical group names, or mesh
selection names—the resolver handles all three transparently.

```python
# Node-to-node: tie beam to columns at shared interface
g.constraints.equal_dof("beam_top", "col_left",  tolerance=1e-3)
g.constraints.equal_dof("beam_top", "col_right", tolerance=1e-3)

# Rigid diaphragm for a floor slab
g.constraints.rigid_diaphragm("slab", "frame",
                               master_point=(3.0, 3.0, 3.5))

# Surface-to-surface tie
g.constraints.tie("beam_top", "slab", tolerance=0.5)
```

Available constraint types span four levels: node-to-node (equal_dof, rigid_link,
penalty), node-to-group (rigid_diaphragm, rigid_body, kinematic_coupling),
node-to-surface (tie, distributing_coupling, embedded), and surface-to-surface
(tied_contact, mortar).

### Resolve constraints

Resolution needs the mesh data and spatial maps:

```python
fem = g.mesh.queries.get_fem_data(dim=3)
node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
face_map = g.parts.build_face_map(node_map)

records = g.constraints.resolve(
    fem.nodes.ids, fem.nodes.coords,
    node_map=node_map, face_map=face_map,
)
```

Each record contains the concrete master/slave node tags, DOFs, and weights
needed by the solver.

## Phase 8 — Solver Export

### OpenSees

```python
g.opensees.build()
g.opensees.export.py("frame_model.py")
# or
g.opensees.export.tcl("frame_model.tcl")
```

### Direct FEM data access

For custom solvers or post-processing:

```python
fem = g.mesh.queries.get_fem_data(dim=3)

fem.nodes.ids              # 1-based contiguous IDs
fem.nodes.coords           # (N, 3) array
fem.elements.ids           # 1-based contiguous IDs
fem.elements.connectivity  # (E, nodes_per_element) array
fem.nodes.physical         # PhysicalGroupSet — query by name or tag
fem.mesh_selection         # MeshSelectionStore — query by name or tag
```

## Closing the Session

```python
g.end()
```

## Complete Example — Portal Frame

```python
from apeGmsh import Part, apeGmsh

# ── Define reusable parts ─────────────────────────────────
column = Part("column")
column.begin()
column.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0)
column.save()
column.end()

beam = Part("beam")
beam.begin()
beam.model.geometry.add_box(0, 0, 0,  6.0, 0.3, 0.5)
beam.save()
beam.end()

# ── Assemble ──────────────────────────────────────────────
g = apeGmsh("portal_frame")
g.begin()

g.parts.add(column, label="col_left",  translate=(0, 0, 0))
g.parts.add(column, label="col_right", translate=(5.5, 0, 0))
g.parts.add(beam,   label="beam_top",  translate=(0, 0, 3.0))

g.parts.fragment_all()

# Physical groups
g.parts.add_physical_groups()

# Mesh
g.mesh.sizing.set_size_global(min_size=50, max_size=150)
g.mesh.generation.generate(dim=3)

# Constraints
g.constraints.equal_dof("beam_top", "col_left",  tolerance=1e-3)
g.constraints.equal_dof("beam_top", "col_right", tolerance=1e-3)

# Resolve and export
fem = g.mesh.queries.get_fem_data(dim=3)
node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
face_map = g.parts.build_face_map(node_map)
g.constraints.resolve(fem.nodes.ids, fem.nodes.coords,
                      node_map=node_map, face_map=face_map)

g.opensees.build()
g.opensees.export.py("portal_frame.py")
g.end()
```

## Summary of the Pipeline

```
Part                          apeGmsh (Assembly)
─────                         ─────────────────
1. Build geometry              3. Import & position parts
2. Save to STEP                4. Fragment interfaces
                               5. Define physical groups
                               6. Generate mesh
                               7. Create mesh selections
                               8. Define & resolve constraints
                               9. Export to solver
```

Part is optional—you can skip it entirely and build geometry inline with
`g.parts.part("name")` or directly through `g.model`.  The Part workflow shines
when you have reusable components (a standard column section, a parametric beam,
a precast panel) that appear multiple times in the model or across projects.
