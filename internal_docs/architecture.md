# apeGmsh Architecture

## What apeGmsh is

apeGmsh is a Python wrapper around [Gmsh](https://gmsh.info/) designed
for structural FEM workflows. It adds what Gmsh lacks: parts-based
assembly, a two-tier naming system that survives boolean operations,
pre-mesh constraint/load/mass definitions that resolve against any
mesh, and a solver-agnostic FEM data broker.

The library is organized as a composition of focused sub-modules,
each owning a narrow responsibility. No inheritance hierarchies, no
god-classes — just composites attached to a session object.


## Session lifecycle

Everything starts with a session:

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="bridge")
g.begin()

# ... geometry, mesh, FEM ...

g.end()
```

Or as a context manager:

```python
with apeGmsh(model_name="bridge") as g:
    # ...
```

`begin()` calls `gmsh.initialize()` + `gmsh.model.add()` and wires
all composites. `end()` calls `gmsh.finalize()`. Gmsh has process-wide
state — only one session can be active at a time.

**Exception:** `Part` objects manage their own isolated Gmsh session.
They `begin()/end()` independently and persist geometry to a STEP
tempfile. The assembly session imports the STEP file — the two
sessions never overlap.


## The composite tree

After `g.begin()`, the session object `g` has 17 composites:

```
g
  |-- .model              Geometry creation, booleans, transforms, I/O, queries
  |     |-- .geometry     Primitives (box, cylinder, sphere, curves, surfaces)
  |     |-- .boolean      fuse, cut, intersect, fragment
  |     |-- .transforms   translate, rotate, scale, mirror, extrude, revolve
  |     |-- .io           STEP/IGES/DXF/MSH load and save, healing
  |     |-- .queries      bbox, COM, mass, boundary, adjacency, registry
  |     +-- .selection    Spatial queries (points in box, surfaces on plane)
  |
  |-- .labels             Internal naming (Tier 1) — survives boolean ops
  |-- .physical           Solver-facing physical groups (Tier 2)
  |-- .parts              Part registry — import, fragment, fuse, node/face maps
  |-- .sections           Parametric section builders (W-beams, rectangles, shells)
  |
  |-- .constraints        Pre-mesh constraint definitions (12 types)
  |-- .loads              Pre-mesh load definitions (7 types, pattern grouping)
  |-- .masses             Pre-mesh mass definitions (4 types)
  |
  |-- .mesh               Meshing pipeline
  |     |-- .generation   generate, set_order, refine, optimize, algorithms
  |     |-- .sizing       Global/per-entity size, callbacks, by-physical
  |     |-- .field        Distance, Threshold, Box, BoundaryLayer, MathEval
  |     |-- .structured   Transfinite curves/surfaces/volumes, recombine
  |     |-- .editing      Embed, periodic, STL import, node relocation
  |     |-- .queries      get_nodes, get_elements, get_fem_data, quality
  |     +-- .partitioning partition, renumber (RCM, Hilbert)
  |
  |-- .loader             MshLoader — standalone .msh file import
  |-- .mesh_selection     Post-mesh named selections (nodes, elements, spatial)
  |-- .partition          Domain decomposition for parallel solvers
  |-- .view               Gmsh view data (post-processing fields)
  |
  |-- .opensees           OpenSees solver bridge
  |     |-- .materials    Material registry (ND, UniAxial, Section)
  |     |-- .elements     Element assignment + geometric transforms
  |     |-- .ingest       Load/mass/constraint injection from FEMData
  |     |-- .inspect      Post-build tables and summaries
  |     +-- .export       TCL/Python script emission
  |
  |-- .inspect            Geometry/mesh summary (DataFrames)
  +-- .plot               Matplotlib plotting (optional)
```


## The four phases

Every apeGmsh workflow follows four phases:

### Phase 1: Geometry

Build or import geometry. Use primitives (`add_box`, `add_cylinder`),
CAD import (`load_step`), or parametric sections (`g.sections.W_solid`).
Apply transforms (`translate`, `rotate`, `extrude`).

Key concept: **entities** are identified by `(dim, tag)` pairs.
`dim=0` is a point, `dim=1` a curve, `dim=2` a surface, `dim=3` a
volume. Tags are integers assigned by Gmsh.

### Phase 2: Assembly & naming

If you have multiple parts, use `g.parts` to register them and
`g.parts.fragment_all()` to make the geometry conformal (shared
faces at interfaces).

Name things with two tiers:
- **Labels** (Tier 1): `g.labels.add(dim, tags, "shaft")` — internal
  bookkeeping. Survive boolean operations via snapshot/remap. Not
  visible to solvers.
- **Physical groups** (Tier 2): `g.physical.add_volume(tags, "Body")` —
  solver-facing declarations. What the solver sees.

Labels are the workhorse. Physical groups are for solver communication.

### Phase 3: Pre-mesh definitions

Define constraints, loads, and masses against names — not against
mesh nodes (which don't exist yet):

```python
g.constraints.equal_dof("slab", "column_top", dofs=[1, 2, 3])

with g.loads.pattern("Gravity"):
    g.loads.gravity("Body", density=2400)

g.masses.volume("Body", density=2400)
```

These are stored as lightweight definition objects. No mesh math yet.

### Phase 4: Mesh + FEM extraction

Generate the mesh, then extract the FEM data broker:

```python
g.mesh.sizing.set_global_size(0.5)
g.mesh.generation.generate(dim=3)

fem = g.mesh.queries.get_fem_data(dim=3)
```

`get_fem_data()` resolves all pre-mesh definitions against the
actual mesh. Constraints become node-pair records. Loads become
per-node force vectors. Masses become per-node mass tuples. Everything
is frozen into `FEMData` — a self-contained broker that needs no live
Gmsh session.


## The FEM data broker

`FEMData` is the bridge between mesher and solver:

```
fem
  |-- .nodes              NodeComposite
  |     |-- .ids          All node IDs (ndarray)
  |     |-- .coords       All coordinates (ndarray)
  |     |-- .get(pg=, label=)  Selection API → NodeResult (iter yields (id, xyz) pairs)
  |     |-- .constraints  Node-pair constraints (equal_dof, rigid, etc.)
  |     |-- .loads        Nodal loads (point forces, face-concentrated)
  |     |-- .sp           SP records (face-prescribed displacements / fix)
  |     +-- .masses       Lumped nodal masses
  |
  |-- .elements           ElementComposite
  |     |-- .ids          All element IDs
  |     |-- .connectivity All connectivity
  |     |-- .get(pg=, label=)  Selection API → ElementResult(ids, conn)
  |     |-- .constraints  Surface constraints (tie, mortar, etc.)
  |     +-- .loads        Element loads (pressure, body force)
  |
  |-- .info               MeshInfo (n_nodes, n_elems, bandwidth)
  +-- .inspect            Summaries, tables, source tracing
```

Construction: `FEMData.from_gmsh(dim=3, session=g)` or
`FEMData.from_msh("file.msh", dim=2)`.


## The two-tier naming system

This is the hardest concept in apeGmsh and the one that makes
multi-part assembly work.

**Problem:** Gmsh physical groups are fragile. When you run a boolean
operation (fragment, fuse, cut), entity tags change. Physical groups
that referenced old tags become stale.

**Solution:** Labels (Tier 1) are stored as internal physical groups
with a `_label:` prefix. Before every boolean operation, apeGmsh
snapshots all label PGs. After the operation, it remaps them to the
new entity tags via the result map. This is invisible to the user.

Physical groups (Tier 2) are the explicit, solver-facing names that
the user creates. They are also preserved through boolean ops via the
same snapshot/remap mechanism.

The user sees:
```python
g.labels.entities("shaft")          # Tier 1: always works
g.physical.add_volume(tags, "Body") # Tier 2: solver sees this
g.labels.promote_to_physical("shaft")  # promote Tier 1 → Tier 2
```


## The Parts system

Parts solve the reuse problem: define geometry once, instantiate many
times with different transforms.

```python
# Define once
col = Part("column")
with col:
    col.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3, label="shaft")

# Instantiate many times
g.parts.add(col, label="col_A", translate=(0, 0, 0))
g.parts.add(col, label="col_B", translate=(5, 0, 0))
g.parts.add(col, label="col_C", translate=(10, 0, 0))

# Fragment to make conformal
g.parts.fragment_all()
```

Behind the scenes:
1. `Part.end()` auto-persists to a STEP tempfile + sidecar JSON
2. `g.parts.add()` imports the STEP into the assembly session
3. Labels are rebound to the imported entities via COM matching
4. Each instance gets prefixed labels: `col_A.shaft`, `col_B.shaft`
5. `fragment_all()` preserves all labels through OCC fragmentation


## The constraint/load/mass pipeline

All three follow the same two-stage pattern:

```
Stage 1 (pre-mesh):     g.constraints.equal_dof("slab", "column", ...)
                         → stores a lightweight ConstraintDef

Stage 2 (post-mesh):     fem = g.mesh.queries.get_fem_data(dim=3)
                         → ConstraintResolver resolves defs against mesh
                         → NodePairRecord objects land in fem.nodes.constraints
```

The resolver needs:
- Node IDs and coordinates (from the mesh)
- Element IDs and connectivity (for face-based constraints)
- Node/face maps (from PartsRegistry, for part-label resolution)

The resolved records are **solver-agnostic**. OpenSees, Abaqus,
Code_Aster — any adapter can consume them. The adapter is a thin
translation layer, not a computational layer.


## The OpenSees bridge

The OpenSees bridge is the reference solver adapter:

```python
g.opensees.set_model(ndm=3, ndf=3)
g.opensees.materials.add_nd_material("Concrete", "ElasticIsotropic",
                                      E=30e9, nu=0.2, rho=2400)
g.opensees.elements.assign("Body", "FourNodeTetrahedron",
                            material="Concrete")
g.opensees.elements.fix("Base", dofs=[1, 1, 1])
g.opensees.build()

g.opensees.export.tcl("model.tcl")
g.opensees.export.py("model.py")
```

The bridge reads from `FEMData` during `build()`. It does NOT call
Gmsh directly — all mesh data comes through the broker.


## The viewer

Two viewers exist:

**ModelViewer** (`g.model.viewer()`) — BRep geometry viewer:
- Entity picking, box select
- Physical group creation/editing
- Load/mass/constraint overlays (when `fem=` provided)
- Parts tree

**MeshViewer** (`g.mesh.viewer()`) — Mesh topology viewer:
- Element/node picking modes
- Node labels, element labels
- Wireframe toggle
- Dimension filtering

Both are Qt/PyVista applications. Optional dependency (`pip install
"apeGmsh[viewer]"`).


## The Results system

Post-processing without a live Gmsh session:

```python
from apeGmsh import Results

r = Results.from_fem(
    fem,
    point_fields={"displacement": disp_array},
    cell_fields={"stress": stress_array},
)
r.to_vtu("output.vtu")
r.viewer()
```

Results wraps VTK export + PyVista visualization. It takes `FEMData`
+ field arrays and produces `.vtu` (single step) or `.pvd` (time
series) files.


## Dependency graph

```
gmsh (C library)
  |
  apeGmsh (core: geometry, labels, parts, constraints, loads, masses)
  |
  +-- mesh module (Gmsh API calls for meshing)
  |     |
  |     +-- FEMData (solver-agnostic output)
  |
  +-- solvers/ (OpenSees bridge)
  |     |
  |     +-- openseespy (optional)
  |
  +-- viz/ (matplotlib, optional)
  |
  +-- viewers/ (PySide6 + PyVista, optional)
  |
  +-- results/ (VTK export)
```

Core dependencies: `gmsh`, `numpy`, `pandas`.
Optional: `matplotlib`, `openseespy`, `PySide6`, `pyvista`, `vtk`.


## File organization

```
src/apeGmsh/
  core/          Geometry, labels, parts, constraints, loads, masses
  mesh/          Meshing pipeline, FEMData, extraction
  solvers/       OpenSees bridge, constraint/load/mass resolvers
  viz/           Lightweight plotting, selection, VTK export
  viewers/       Qt interactive viewers (30 files)
  sections/      Parametric section builders
  results/       Post-processing
```

Every composite is a thin class in one file. Sub-composites are
prefixed with `_` (private modules). Public API is exported through
`__init__.py`.


## Design principles

1. **Composition over inheritance** — the session is a container,
   not a base class. Composites don't inherit from each other.

2. **Names survive operations** — labels and physical groups are
   preserved through every boolean operation, every remesh, every
   part import. Names are the only stable identifier.

3. **Define before mesh, resolve after** — constraints, loads, and
   masses are symbolic until the mesh exists. This decouples the
   engineering intent from the mesh realization.

4. **The broker is the boundary** — `FEMData` is the single point
   where live Gmsh state becomes frozen data. Everything downstream
   (solvers, viewers, results) works from the broker, never from Gmsh.

5. **Solver adapters are thin** — the resolver does the heavy math
   (tributary integration, surface projection, interpolation weights).
   The solver adapter just translates records to commands.

6. **Optional dependencies are lazy** — matplotlib, openseespy,
   PySide6, pyvista are imported at call time, not at module level.
   `pip install apeGmsh` gives you the full core with no heavy deps.
