# Parts vs. Direct Session: When to Use Each

pyGmsh gives you two ways to build geometry for a finite element model. You can
define reusable **Parts** that get imported into a session, or you can build
everything **directly inside a single pyGmsh session**. Both paths end at the
same place — a meshed, solver-ready model — but they suit different problems.

This guide explains the tradeoffs, then digs into what actually happens to the
mesh when multiple bodies meet: fragmentation, conformal interfaces, and what
goes wrong when you skip it.


## Two Approaches

### Approach A — Isolated Parts

Each component is built in its own Gmsh session, exported to STEP, and imported
into the assembly session.

```python
from apeGmsh import Part, pyGmsh

# Build in isolation
col = Part("column")
col.begin()
col.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0)
col.save()
col.end()

# Assemble
g = apeGmsh("frame")
g.begin()
g.parts.add(col, label="col_1", translate=(0, 0, 0))
g.parts.add(col, label="col_2", translate=(6, 0, 0))
g.parts.fragment_all()
g.mesh.generate(dim=3)
```

### Approach B — Direct Session

Everything is created inside one pyGmsh session. You can optionally wrap
geometry in `g.parts.part()` blocks to give it a name and track it for
fragmentation and constraints.

```python
from apeGmsh import apeGmsh

g = apeGmsh("frame")
g.begin()

with g.parts.part("col_1"):
    g.model.geometry.add_box(0, 0, 0,  0.5, 0.5, 3.0)

with g.parts.part("col_2"):
    g.model.geometry.add_box(6, 0, 0,  0.5, 0.5, 3.0)

g.parts.fragment_all()
g.mesh.generate(dim=3)
```

Or, at its simplest, skip `g.parts` entirely:

```python
g = apeGmsh("plate")
g.begin()
g.model.geometry.add_rectangle(0, 0, 0,  10, 5)
g.mesh.generate(dim=2)
```


## When to Use Each

### Use Parts when...

**You have reusable components.** A standard column section, a parametric beam,
a precast panel. Define it once as a Part, save the STEP, and import it multiple
times with different positions. The geometry is authored once and the STEP file
becomes a portable artifact that can be version-controlled or shared across
projects.

**Geometry comes from external CAD.** If someone hands you a SolidWorks or
FreeCAD STEP file, `g.parts.import_step()` brings it in without touching
the original. The Part class itself is optional here — you can import STEP files
directly — but the mental model is the same: geometry is defined elsewhere,
assembly happens in pyGmsh.

**You want clean separation of concerns.** Part sessions cannot mesh, cannot
define physical groups, cannot set constraints. This is by design. It forces
all solver-facing decisions into the assembly session, which means you can
change mesh density or constraint strategy without touching the geometry
definition.

**Multiple people work on the model.** One person defines the column geometry,
another defines the slab. Each saves a STEP file. The assembly script imports
both and handles positioning, meshing, and coupling.

### Use a direct session when...

**The model is a single body.** A plate with a hole, a circular annulus, a
cantilever beam. There are no interfaces between parts, no fragmentation needed,
and no reason to save intermediate STEP files.

**You need fine-grained meshing control.** Transfinite meshing, structured
grids, and recombined quads often require careful coordination between the
geometry and the meshing calls. Doing everything in one session — building
geometry, then immediately setting transfinite constraints on the curves and
surfaces you just created — is more natural than building in a Part, exporting,
re-importing, and then hunting for the right entity tags.

**You are running parametric sweeps.** In a convergence study or parameter scan,
you rebuild geometry from scratch each iteration. The overhead of creating a
Part, saving to disk, and importing adds no value when the geometry is
disposable.

**Rapid prototyping.** When you are exploring a geometry idea and iterating
quickly, the single-session approach has less boilerplate.


## Comparison at a Glance

```
                         Parts + Import         Direct Session
                         ──────────────         ──────────────
Reusability              High (STEP files)      None (inline)
External CAD             Natural fit            Possible via g.model.io.load_step()
Session isolation        Yes (own Gmsh env)     No (shared session)
Boilerplate              Higher                 Lower
Transfinite control      After re-import        Immediate
Parametric sweeps        Overkill               Ideal
Multi-person workflow    Clean handoff           Harder
Fragmentation needed     Almost always           Only if multi-body
```


## What Happens to the Mesh: Fragmentation and Coherence

This is the key question for any multi-body model. When you place two volumes
next to each other in Gmsh, they are independent geometric entities. Gmsh meshes
each one separately. Unless you do something about it, the nodes at their shared
interface will **not** line up.

### The Problem: Non-Conformal Interfaces

Imagine two boxes touching along a face. Without fragmentation, Gmsh sees two
separate volumes, each with its own copy of the shared surface. It meshes them
independently:

```
  Box A                Box B
  ┌──┬──┬──┐          ┌───┬───┐
  │  │  │  │          │   │   │
  ├──┼──┼──┤          ├───┼───┤
  │  │  │  │          │   │   │
  ├──┼──┼──┤          ├───┼───┤
  │  │  │  │          │   │   │
  └──┴──┴──┘          └───┴───┘
        ↑  ↑
    These nodes don't match
```

Box A might put 4 nodes on the interface. Box B might put 3. The nodes are at
different positions. There is no direct node-to-node connection between the two
meshes. For a finite element solver, this means the interface either has gaps in
the displacement field or requires special treatment (mortar elements, penalty
coupling, tied contact).

### The Solution: Boolean Fragmentation

`g.parts.fragment_all()` calls `gmsh.model.occ.fragment()` on all tracked
entities. This is a boolean operation that **splits the geometry at every
intersection**, producing a single conformal topology where shared interfaces
become shared geometric entities.

After fragmentation:

```
  Box A        Shared       Box B
  ┌──┬──┬──┐──┬──┬──┐──┬───┬───┐
  │  │  │  │  │  │  │  │   │   │
  ├──┼──┼──┼──┼──┼──┼──┼───┼───┤
  │  │  │  │  │  │  │  │   │   │
  ├──┼──┼──┼──┼──┼──┼──┼───┼───┤
  │  │  │  │  │  │  │  │   │   │
  └──┴──┴──┴──┴──┴──┴──┴───┴───┘
              ↑
     Shared surface: one set of nodes
```

The interface surface now belongs to both volumes. When Gmsh meshes, nodes on
that surface are shared. The two bodies are connected node-to-node with no gaps,
no overlaps, and no special constraints needed.

### What Fragmentation Actually Does

Internally, `fragment_all()` performs these steps:

1. **Collects all tracked entities** at the highest dimension present (3D volumes,
   or 2D surfaces, or 1D curves).

2. **Calls `gmsh.model.occ.fragment()`** — the OCC boolean engine computes all
   intersections and splits every entity at every crossing. A volume that was one
   piece may become several pieces if other volumes cut through it.

3. **Builds an old-tag to new-tag map.** After fragmentation, entity tags change.
   What was volume tag 1 might now be volumes 5, 6, and 7 (because two other
   volumes cut through it). The `result_map` from Gmsh tells us exactly which
   new tags came from which old tags.

4. **Updates each Instance's entity list in-place.** This is critical — without
   it, the part labels would point to tags that no longer exist. After this step,
   `g.parts.get("col_1").entities` contains the correct post-fragment tags.

5. **Synchronizes the OCC kernel** so the new topology is available for meshing.

### What Can Go Wrong

**Orphaned entities.** If you create geometry outside of `g.parts.part()` blocks
and don't register it, `fragment_all()` will warn you. The orphan entities still
participate in fragmentation (they are in the Gmsh model), but their new tags
won't be tracked by any instance. This can cause confusion later when building
node maps for constraints. Fix: use `g.parts.from_model()` or
`g.parts.register()` to adopt orphans before fragmenting.

**Tag invalidation.** Any entity tag you saved before calling `fragment_all()`
is potentially invalid afterward. Don't cache tags across a fragment call. Always
query fresh tags from `g.parts.get(label).entities` or from physical groups
after fragmentation.

**Topological surprises.** Fragmentation can create new entities you didn't
expect. Two cubes touching along an edge will produce shared edge and vertex
entities. A beam volume passing through a slab volume will split the slab into
multiple pieces. The total entity count after fragmentation is often larger than
before. This is correct behavior — it's the geometric price of conformal
interfaces.

**Near-miss geometry.** If two bodies are *almost* touching but have a tiny gap
(say, 1e-8 mm due to floating-point), the OCC kernel may not detect the
intersection. The result: no fragmentation at that interface, non-conformal mesh,
and potentially mysterious solver errors. Cure: make sure your geometry is
built with exact coincidence at interfaces, or use `remove_duplicates()` with
a tolerance before fragmenting.


## When You Don't Fragment

Not every multi-body model needs fragmentation. There are legitimate cases where
you **want** non-conformal interfaces:

**DOF-incompatible element coupling.** When coupling solid elements (3 translational
DOFs per node) with beam/frame elements (6 DOFs per node), you cannot share
nodes directly — the solid nodes lack rotational stiffness. The standard approach
is to keep the meshes separate and use constraints: duplicated interface nodes
with `rigidLink` for kinematics and `equalDOF` for translational ties.

**Contact problems.** If two bodies can slide or separate, their meshes must be
independent. Contact algorithms (penalty, Lagrange multiplier, augmented
Lagrangian) handle the interface coupling at the solver level.

**Domain reduction methods.** In soil-structure interaction, the near-field and
far-field may be meshed independently with coupling through DRM forces at
the boundary.

**Multi-scale analysis.** A coarse global model and a fine local model may
overlap in space but are solved separately with boundary conditions passed
between them.

In all these cases, you still use `g.parts` to track the bodies and build node
maps, but you skip `fragment_all()` and rely on constraints instead:

```python
g.parts.add(solid_part, label="solid")
g.parts.add(frame_part, label="frame")
# No fragment_all() — meshes are independent

g.mesh.generate(dim=3)

g.constraints.equal_dof("solid", "frame", dofs=[1, 2, 3], tolerance=1e-3)
g.constraints.rigid_link("frame", "solid", link_type="beam")

fem = g.mesh.get_fem_data(dim=3)
node_map = g.parts.build_node_map(fem.node_ids, fem.node_coords)
records = g.constraints.resolve(fem.node_ids, fem.node_coords, node_map=node_map)
```


## The Middle Ground: `g.model.queries.make_conformal()`

For single-session models (no Parts, no `g.parts`), there is a lower-level
fragmentation path on the Model composite:

```python
g = apeGmsh("multi_region")
g.begin()
g.model.geometry.add_box(0, 0, 0,  1, 1, 1)
g.model.geometry.add_box(1, 0, 0,  1, 1, 1)
g.model.queries.make_conformal()    # fragments all entities at their intersections
g.mesh.generate(dim=3)
```

`g.model.queries.make_conformal()` (aliased as `g.model.queries.fragment_all()`) performs the
same OCC boolean fragmentation but without instance tracking. Use it when you
have multiple geometric regions in one session but haven't bothered with
`g.parts`. The tradeoff: you won't have labeled instances, so building node maps
and constraints is manual.


## Geometric Construction: Composing One Body from Multiple Primitives

There is a third pattern that sits between "one Part = one primitive" and "just
build everything in the session." Sometimes you need to compose a complex shape
from simpler geometric operations — an I-beam from three boxes, an L-shaped wall
from two rectangles, a foundation with cutouts. The result is conceptually
**one body** with one material, one set of properties, and no internal interfaces.

This is geometric construction, not assembly. The difference matters for the mesh:
you don't want conformal interfaces inside the I-beam web-flange junction — you
want those internal surfaces to disappear entirely so the mesher treats it as a
single volume.

### At the Part level

The simplest approach: build and fuse inside a Part's isolated session using
`model.fuse()`. The fused result saves to one STEP file and imports as one
Instance.

```python
from apeGmsh import Part

beam = Part("i_beam")
beam.begin()

web        = beam.model.geometry.add_box(0, 0, 0,       0.01, 0.3, 5.0)
flange_bot = beam.model.geometry.add_box(-0.1, -0.005, 0, 0.2, 0.005, 5.0)
flange_top = beam.model.geometry.add_box(-0.1, 0.295, 0,  0.2, 0.005, 5.0)

beam.model.boolean.fuse([web], [flange_bot, flange_top])

beam.properties["material"] = "steel"
beam.properties["section"]  = "W360x33"
beam.save()
beam.end()
```

After `fuse()`, the three boxes become one volume. The surfaces where the flanges
meet the web are gone — the mesher sees a single solid. This is the recommended
approach when the composed body will be reused or imported into an assembly.

### At the session level with `fuse_group()`

When you are building geometry inline and don't need a separate Part, you can
define the primitives as tracked parts and then fuse them:

```python
g = apeGmsh("frame")
g.begin()

with g.parts.part("web"):
    g.model.geometry.add_box(0, 0, 0,  0.01, 0.3, 5.0)
with g.parts.part("flange_bot"):
    g.model.geometry.add_box(-0.1, -0.005, 0,  0.2, 0.005, 5.0)
with g.parts.part("flange_top"):
    g.model.geometry.add_box(-0.1, 0.295, 0,  0.2, 0.005, 5.0)

g.parts.fuse_group(["web", "flange_bot", "flange_top"], label="i_beam")
```

`fuse_group()` calls OCC fuse on the listed instances, removes the old labels
from the registry, and creates a single new Instance under the given label. The
internal surfaces vanish. Subsequent `fragment_all()` against other parts will
treat the I-beam as one body.

### Fuse vs. Fragment — When to Use Which

The distinction comes down to whether the internal boundaries should exist in the
mesh:

**Fuse** erases internal surfaces. Use it when the primitives are construction
aids for one physical body. After fusing, there is no interface — the volume is
monolithic. The mesher generates elements freely across what used to be the
web-flange junction.

**Fragment** preserves internal surfaces as shared boundaries. Use it when the
parts represent different physical regions (different materials, different element
types, different mesh densities). After fragmenting, each region is a separate
volume that shares nodes at the interface.

```
Fuse (geometric construction)     Fragment (assembly)
─────────────────────────────     ────────────────────
Input:  3 boxes (web + flanges)   Input:  beam + column
Result: 1 volume, no internal     Result: 2 volumes, shared interface
        surfaces                          surface with shared nodes
Use:    One material, one body    Use:    Different materials/regions
```

A typical workflow combines both: fuse primitives into bodies, then fragment the
bodies against each other.

```python
# Build two composite parts
g.parts.fuse_group(["web", "fl_top", "fl_bot"], label="beam")
g.parts.fuse_group(["col_core", "col_cover"], label="column")

# Now fragment them for conformal meshing at the beam-column interface
g.parts.fragment_all()
g.mesh.generate(dim=3)
```


## Decision Flowchart

```
Is the model a single body?
  ├── Yes → Direct session, no fragmentation needed
  └── No → Multiple bodies
              │
              Do the bodies share interfaces that need conformal nodes?
              ├── Yes → Use g.parts + fragment_all()
              │         (or g.model.queries.make_conformal() for quick models)
              └── No → Are they DOF-incompatible or in contact?
                        ├── Yes → Use g.parts without fragment,
                        │         couple via constraints
                        └── No → They are independent models,
                                  consider separate pyGmsh sessions
```


## Summary

Parts give you reusability and clean separation at the cost of an extra
save/import step. Direct sessions give you speed and immediate control at the
cost of everything being tangled in one place. For most structural analysis
workflows with multiple components, the Part approach plus `fragment_all()` is
the right default — it produces conformal meshes where bodies share interface
nodes, which is what standard FEM solvers expect. When the physics demands
non-conformal coupling (contact, DOF-incompatible elements, multi-scale), skip
fragmentation and use the constraint system instead.
