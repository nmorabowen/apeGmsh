# Gmsh Geometry Basics

How geometry is created, modified, and how `(dim, tag)` tuples flow through operations.

---

## Bottom-Up Construction

The fundamental way to build geometry in Gmsh is bottom-up: start at dim 0 and work your way up. This is the native workflow of the `geo` kernel, and the mental model even when using OCC shortcuts.

```
Points (0)  →  Curves (1)  →  Curve Loops  →  Surfaces (2)  →  Surface Loops  →  Volumes (3)
```

Each step returns the **tag** of the created entity. You control tags explicitly (pass `tag=N`) or let Gmsh auto-assign (pass `tag=-1`, the default).

```python
# geo kernel — fully explicit
p1 = gmsh.model.geo.addPoint(0, 0, 0)       # → tag 1
p2 = gmsh.model.geo.addPoint(1, 0, 0)       # → tag 2
p3 = gmsh.model.geo.addPoint(1, 1, 0)       # → tag 3
p4 = gmsh.model.geo.addPoint(0, 1, 0)       # → tag 4

l1 = gmsh.model.geo.addLine(p1, p2)         # → tag 1
l2 = gmsh.model.geo.addLine(p2, p3)         # → tag 2
l3 = gmsh.model.geo.addLine(p3, p4)         # → tag 3
l4 = gmsh.model.geo.addLine(p4, p1)         # → tag 4

cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])  # → tag 1
s  = gmsh.model.geo.addPlaneSurface([cl])            # → tag 1

gmsh.model.geo.synchronize()
```

Every integer in an API call that expects entity references is a **tag** — the unique identifier within that dimension. `addLine(p1, p2)` means "create a line from point with tag `p1` to point with tag `p2`." Tags are always integers, never coordinates.

**Tag tracking is trivial here.** Every tag is returned directly, tags are stable, and nothing changes until you explicitly create something new.

The OCC kernel supports the same bottom-up workflow with the same function names under `gmsh.model.occ.*`. The difference: OCC auto-assigns tags internally and they become queryable only after `synchronize()`.

### Curve loops — ordering and orientation

A curve loop defines a closed boundary for a surface. The curves must form a **connected, closed chain**: the end point of each curve must coincide with the start point of the next.

```python
cl = gmsh.model.geo.addCurveLoop([1, 2, 3, -4])
#                                 ^  ^  ^  ^^
#                                 │  │  │  │└─ curve tag 4
#                                 │  │  │  └── negative = reversed direction
#                                 │  │  └───── curve tag 3, forward
#                                 │  └──────── curve tag 2, forward
#                                 └─────────── curve tag 1, forward
```

Each number is a **curve tag**. The **sign** encodes traversal direction: positive means the curve is followed from its start point to its end point, negative means reversed (end → start). This is how you stitch curves that were defined in different directions into a continuous loop.

**Ordering matters.** The curves must be listed so that they chain head-to-tail:

```
curve 1:  p1 ──► p2
curve 2:  p2 ──► p3
curve 3:  p3 ──► p4
curve -4: p1 ◄── p4  (curve 4 defined as p4→p1, reversed to go p4→p1... 
                       wait — reversed means we traverse p1→p4 backwards,
                       so we enter at p4 and exit at p1? No:
                       curve 4 was defined addLine(p4, p1), so
                       +4 goes p4→p1, -4 goes p1→p4.
                       But we need p4→p1 to close the loop, so we use +4)
```

Let's be precise. Given:

```python
l1 = addLine(p1, p2)   # +l1: p1→p2,  -l1: p2→p1
l2 = addLine(p2, p3)   # +l2: p2→p3,  -l2: p3→p2
l3 = addLine(p3, p4)   # +l3: p3→p4,  -l3: p4→p3
l4 = addLine(p4, p1)   # +l4: p4→p1,  -l4: p1→p4
```

A valid loop: `[l1, l2, l3, l4]` → p1→p2→p3→p4→p1 (counterclockwise if points are CCW).

Also valid: `[-l4, -l3, -l2, -l1]` → p1→p4→p3→p2→p1 (same loop, opposite winding).

**What happens if curves are out of order?** In the `geo` kernel, it will fail — Gmsh expects an explicitly ordered chain. In the OCC kernel, `addCurveLoop` will **attempt to reorder** the curves automatically (it finds a valid chain from the unordered set). But relying on auto-reordering is fragile with complex geometry; explicit ordering is safer.

**Winding direction:** counterclockwise defines a surface whose normal points outward (right-hand rule). Clockwise gives an inward normal. For the outer boundary of a surface, CCW is conventional. For holes, the winding is typically opposite (CW), but Gmsh handles this when you pass holes as separate wire arguments.

### Surface loops — ordering

A surface loop defines a closed shell bounding a volume. Same principle: the surfaces must form a closed, watertight boundary.

```python
sl = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
vol = gmsh.model.geo.addVolume([sl])
```

Each number is a **surface tag**. Surface normals should point outward consistently, but the OCC kernel handles orientation automatically. The `geo` kernel is stricter — normals must be consistent.

### Surfaces with holes

For `addPlaneSurface`, the first argument is a list of curve loop tags. The first loop is the **outer boundary**, the rest are **holes**:

```python
outer = gmsh.model.geo.addCurveLoop([...])    # outer boundary
hole1 = gmsh.model.geo.addCurveLoop([...])    # first hole
hole2 = gmsh.model.geo.addCurveLoop([...])    # second hole

s = gmsh.model.geo.addPlaneSurface([outer, hole1, hole2])
#                                   ^^^^^  ^^^^^  ^^^^^
#                                   outer  holes...
```

---

## OCC Solid Primitives

The OCC kernel provides high-level constructors that skip the bottom-up steps entirely. One call produces a complete BRep solid.

```python
box   = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)         # → volume tag
#                              ^^^^^^   ^^^^^^^^^^
#                              origin   dimensions (not tags — these are coordinates/lengths)

cyl   = gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r) # → volume tag
#                                   ^^^^^^   ^^^^^^^^^^   ^
#                                   origin   axis vector  radius

sph   = gmsh.model.occ.addSphere(xc, yc, zc, r)            # → volume tag
cone  = gmsh.model.occ.addCone(x, y, z, dx, dy, dz, r1, r2)# → volume tag
torus = gmsh.model.occ.addTorus(x, y, z, R, r)              # → volume tag
wedge = gmsh.model.occ.addWedge(x, y, z, dx, dy, dz)       # → volume tag
```

> [!note]
> Unlike `addLine(tag, tag)` where arguments are entity tags, primitive constructors take **geometric parameters** (coordinates, lengths, radii). The only tag involved is the return value.

**Tag tracking:** you get back the **top-level** entity tag (the volume). The internal BRep entities — faces, edges, vertices — get auto-assigned tags that are only visible after `synchronize()`.

```python
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)   # → volume tag, e.g. 1
gmsh.model.occ.synchronize()

# Now the internal entities exist and can be queried
faces = gmsh.model.getBoundary([(3, box)])         # → [(2, 1), (2, 2), ... (2, 6)]
edges = gmsh.model.getBoundary([(3, box)], recursive=True)  # → all edges and points
```

> [!note]
> You do **not** control the tags of internal entities. After `synchronize()`, use entity queries (`getBoundary`, `getAdjacencies`, `getEntitiesInBoundingBox`) to discover them.

### Tag variables and what invalidates them

When you store a tag in a Python variable, that variable is just an integer. `synchronize()` does **not** invalidate it — it only makes entities visible to queries, meshing, and physical groups. Your variables remain valid:

```python
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)   # box = 1
cyl = gmsh.model.occ.addCylinder(5, 0, 0, 0, 0, 1, 0.5)  # cyl = 2
gmsh.model.occ.synchronize()
# box is still 1, cyl is still 2 — synchronize doesn't touch tags
```

What **does** invalidate tag variables:

- **Boolean operations** — input entities are destroyed (if `removeObject/removeTool=True`). The variable still holds the old integer, but the entity behind it no longer exists or has been replaced by new entities with different tags.
- **Healing** (`healShapes`) — may split, merge, or recreate entities with new tags.
- **`removeAllDuplicates()`** — merges coincident entities, survivors keep unpredictable tags.

```python
box = gmsh.model.occ.addBox(...)       # box = 1
cyl = gmsh.model.occ.addCylinder(...)  # cyl = 2

out, out_map = gmsh.model.occ.cut([(3, box)], [(3, cyl)])
gmsh.model.occ.synchronize()

# box still holds integer 1, but does (3, 1) still exist?
# Maybe — depends on OCCBooleanPreserveNumbering heuristic.
# cyl (the tool) is definitely gone.
# out_map is the only reliable answer.
```

> [!important]
> After any destructive operation, treat your old tag variables as **stale**. Use the return values (`outDimTags`, `outDimTagsMap`) or entity queries to get the current state. The variable didn't change — the model underneath it did.

### OCC 2D primitives

OCC also provides 2D shape constructors:

```python
rect = gmsh.model.occ.addRectangle(x, y, z, dx, dy)   # → surface tag
disk = gmsh.model.occ.addDisk(xc, yc, zc, rx, ry)     # → surface tag
```

These return surface tags. Their bounding curves/points are auto-generated.

---

## Extrusion and Revolution

Extrusion generates entities one dimension higher by sweeping along a direction. Revolution does the same along a rotational axis.

```python
outDimTags = gmsh.model.occ.extrude(dimTags, dx, dy, dz)
outDimTags = gmsh.model.occ.revolve(dimTags, x, y, z, ax, ay, az, angle)
```

### Return value convention

The return `outDimTags` is a flat list of `(dim, tag)` tuples. **For each input entity**, the output contains:

1. **`outDimTags[0]`** — the "top" entity (same dimension as input, at the far end of the sweep)
2. **`outDimTags[1]`** — the swept volume/surface (one dimension higher than input)
3. **`outDimTags[2:]`** — the lateral entities (same dimension as the swept entity's boundary)

For a surface extruded into a volume:

```python
ov = gmsh.model.occ.extrude([(2, 1)], 0, 0, 1.0)

# ov[0] = (2, N)  ← top surface (copy of input at z=1)
# ov[1] = (3, M)  ← the new volume
# ov[2] = (2, ?)  ← lateral surface (from edge 1 of input)
# ov[3] = (2, ?)  ← lateral surface (from edge 2 of input)
# ...
```

```
        ov[0] ← top face
       ┌──────────┐
      /│          /│
     / │  ov[1]  / │  ← volume
    /  │  (vol) /  │
   ┌──────────┐  ov[3] ← lateral
   │ ov[2]    │   │
   │   └──────│───┘
   │  /       │  /     ← input face (2, 1) at bottom
   │ /        │ /
   └──────────┘
```

**Tag tracking:** the input entity's tag may or may not survive — it depends on the kernel and options. The returned list is the authoritative record of what was created. Always capture it.

### Mesh extrusion

Both `extrude` and `revolve` accept optional parameters to extrude the mesh simultaneously:

```python
ov = gmsh.model.occ.extrude(
    [(2, 1)], 0, 0, h,
    numElements=[8, 2],    # 8 elements in first layer, 2 in second
    heights=[0.5, 1.0],    # cumulative heights, normalized to 1
    recombine=True          # hex/prism instead of tet
)
```

### Multiple input entities

When extruding multiple entities, the return list contains the outputs for **each input in sequence**. For $n$ input entities, the pattern repeats $n$ times, each block starting with the top entity.

```python
ov = gmsh.model.occ.extrude([(2, 1), (2, 2)], 0, 0, 1.0)
# First block:  top of (2,1), volume from (2,1), laterals of (2,1)
# Second block: top of (2,2), volume from (2,2), laterals of (2,2)
```

> [!warning]
> The number of lateral entities per input depends on the input's boundary (how many edges/faces it has). There's no fixed stride. If you need to parse the output for multiple inputs, group by dimension or use entity queries after synchronize.

---

## Boolean Operations

Boolean operations are OCC-only. They create, destroy, and remap entities — this is where tag tracking becomes critical.

All four operations share the same signature:

```python
outDimTags, outDimTagsMap = gmsh.model.occ.fuse(objectDimTags, toolDimTags)
outDimTags, outDimTagsMap = gmsh.model.occ.cut(objectDimTags, toolDimTags)
outDimTags, outDimTagsMap = gmsh.model.occ.intersect(objectDimTags, toolDimTags)
outDimTags, outDimTagsMap = gmsh.model.occ.fragment(objectDimTags, toolDimTags)
```

And the same optional flags:

```python
removeObject=True   # delete original object entities (default)
removeTool=True     # delete original tool entities (default)
tag=-1              # force output tag (only if single result entity)
```

### The two return values

**`outDimTags`** — flat list of all surviving `(dim, tag)` entities after the operation.

**`outDimTagsMap`** — the **parent→child mapping**. This is a list of lists, indexed parallel to the concatenation `objectDimTags + toolDimTags`. For each input entity, it tells you which output entities it became.

```python
input_dimtags = objectDimTags + toolDimTags

# outDimTagsMap[i] = list of (dim, tag) that input_dimtags[i] became
for old_dt, new_dts in zip(input_dimtags, outDimTagsMap):
    print(f"{old_dt} → {new_dts}")
```

This map is the **only reliable way** to track tags through booleans. Input tags are destroyed (if `removeObject/removeTool=True`), output tags are new. The map connects old to new.

### `fuse` — union

Merges objects and tools into a single entity. Shared boundaries are removed.

```python
box1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)  # (3, 1)
box2 = gmsh.model.occ.addBox(0.5, 0, 0, 1, 1, 1)  # (3, 2)

out, out_map = gmsh.model.occ.fuse([(3, 1)], [(3, 2)])
# out     = [(3, 1)]       ← single merged volume
# out_map = [[(3, 1)],     ← (3,1) → (3,1)  object survived (renumbered to itself)
#            [(3, 1)]]     ← (3,2) → (3,1)  tool absorbed into object
```

```
  ┌─────┬─────┐           ┌───────────┐
  │     │     │   fuse    │           │
  │ (3,1)│(3,2)│  ────►   │   (3,1)   │
  │     │     │           │           │
  └─────┴─────┘           └───────────┘
```

### `cut` — difference

Subtracts tools from objects. The tool volume is consumed.

```python
plate = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 1)   # (3, 1)
hole  = gmsh.model.occ.addCylinder(5, 5, 0, 0, 0, 1, 2)  # (3, 2)

out, out_map = gmsh.model.occ.cut([(3, 1)], [(3, 2)])
# out     = [(3, 1)]       ← plate with hole
# out_map = [[(3, 1)],     ← object survived (modified)
#            []]            ← tool consumed (empty — no surviving entity)
```

```
  ┌──────────┐             ┌─────⌢────┐
  │          │             │    / \    │
  │  (3,1)  ○│  cut       │   │   │   │
  │    (3,2) │  ────►     │    \_/    │
  │          │             │   (3,1)   │
  └──────────┘             └──────────┘
```

### `intersect` — common part

Keeps only the volume shared between objects and tools.

```python
out, out_map = gmsh.model.occ.intersect([(3, 1)], [(3, 2)])
# out     = [(3, N)]       ← the intersection volume
# out_map = [[(3, N)],     ← object → intersection
#            [(3, N)]]     ← tool → intersection
```

### `fragment` — conforming split

This is the most important boolean for structural analysis. It splits all entities at their mutual intersections, producing **conforming interfaces** — shared surfaces between adjacent volumes get a single set of mesh nodes.

```python
box  = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 1)    # (3, 1)
box2 = gmsh.model.occ.addBox(1, 0, 0, 2, 1, 1)    # (3, 2)

out, out_map = gmsh.model.occ.fragment([(3, 1)], [(3, 2)])
# out     = [(3, 1), (3, 2), (3, 3)]  ← three volumes: left, overlap, right
# out_map = [[(3, 1), (3, 2)],        ← original box1 split into 2 pieces
#            [(3, 2), (3, 3)]]         ← original box2 split into 2 pieces
```

```
  ┌─────┬─────┐─────┐
  │     │/////│     │
  │(3,1)│(3,2)│(3,3)│     (3,2) is the shared overlap region
  │     │/////│     │
  └─────┴─────┘─────┘
```

> [!important]
> `fragment` is essential for multi-material models. Without it, two adjacent volumes meshed independently will have **duplicate nodes** at the interface — no structural continuity. Fragment ensures a single conformal interface.

### Tag preservation heuristic

When `removeObject=True` and `removeTool=True` (default), Gmsh tries to reuse input tags for output entities when the mapping is simple (controlled by `Geometry.OCCBooleanPreserveNumbering`). But this is a **heuristic**, not a guarantee. Always use `outDimTagsMap` for reliable tracking.

### Tracking labels through booleans

Physical groups reference entity tags. When booleans destroy and recreate entities, those references break. The fix is to snapshot physical groups before the boolean, then remap using `outDimTagsMap`:

```python
# Pattern: snapshot → boolean → remap
input_dimtags = obj_dt + tool_dt

# 1. Snapshot PGs
snapshot = capture_physical_groups()

# 2. Boolean
result, result_map = gmsh.model.occ.fragment(obj_dt, tool_dt)
gmsh.model.occ.synchronize()

# 3. Build old→new mapping and recreate PGs
dt_map = {}
for old_dt, new_dts in zip(input_dimtags, result_map):
    dt_map[old_dt] = new_dts

# For each PG, replace old entity tags with their new equivalents
for pg in snapshot:
    new_tags = []
    for old_tag in pg.entity_tags:
        old_dt = (pg.dim, old_tag)
        if old_dt in dt_map:
            new_tags.extend(t for d, t in dt_map[old_dt] if d == pg.dim)
    gmsh.model.addPhysicalGroup(pg.dim, new_tags, name=pg.name)
```

> [!note]
> apeGmsh automates this pattern in `Labels.remap_physical_groups()` and the `pg_preserved()` context manager.

---

## Transformations

Transformations modify entities **in-place** — the `(dim, tag)` stays the same, only the geometry changes.

```python
gmsh.model.occ.translate(dimTags, dx, dy, dz)
gmsh.model.occ.rotate(dimTags, x, y, z, ax, ay, az, angle)
gmsh.model.occ.mirror(dimTags, a, b, c, d)     # plane ax+by+cz+d=0
gmsh.model.occ.dilate(dimTags, x, y, z, a, b, c)  # scale
```

**Tag tracking is trivial** — tags don't change. The entity at `(dim, tag)` before is the same entity after, just repositioned.

### Copy

`copy` is the exception — it creates **new** entities with new tags:

```python
new_dimtags = gmsh.model.occ.copy(dimTags)
# new_dimtags[i] corresponds to dimTags[i], same order
```

The mapping is positional: `new_dimtags[i]` is the copy of `dimTags[i]`. The copied entity is an independent BRep with no link to the original.

```python
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)       # (3, 1)
copies = gmsh.model.occ.copy([(3, 1)])                # → [(3, 2)]
gmsh.model.occ.translate(copies, 2, 0, 0)             # move the copy, original stays
```

### Remove

`remove` deletes entities:

```python
gmsh.model.occ.remove(dimTags, recursive=False)
# recursive=True also removes all bounding entities (edges, points, etc.)
```

Tags of removed entities are freed and may be reused by subsequent operations.

### `removeAllDuplicates`

Merges coincident entities (same geometric location within tolerance):

```python
gmsh.model.occ.removeAllDuplicates()
```

This can change tags unpredictably — entities that were duplicates get merged into one, and the survivor's tag is implementation-dependent. Use entity queries after calling this.

---

## CAD Import and Healing

### Import

```python
dimTags = gmsh.model.occ.importShapes(fileName, highestDimOnly=True)
```

Supports STEP (`.step`, `.stp`), IGES (`.iges`, `.igs`), and BREP (`.brep`).

**Tag tracking:** `importShapes` returns the `(dim, tag)` list of imported entities. With `highestDimOnly=True` (default), only the top-level entities are returned. Set it to `False` to get everything.

Tags are assigned sequentially starting from the next available tag at each dimension. If you import into a model that already has entities, the new tags won't collide — Gmsh handles that.

### Healing

CAD files from external tools often have geometric defects — tiny edges, degenerate faces, gaps between surfaces. These cause meshing failures.

```python
gmsh.model.occ.healShapes(
    dimTags=[],           # empty = heal everything
    tolerance=1e-8,
    fixDegenerated=True,
    fixSmallEdges=True,
    fixSmallFaces=True,
    sewFaces=True,
    makeSolids=True
)
```

> [!warning]
> Healing can **change tags**. It may split, merge, or recreate entities. After `healShapes()` + `synchronize()`, re-query the model with `getEntities()` to discover the current state. Don't assume tags from `importShapes` are still valid.

### Common import workflow

```python
dimTags = gmsh.model.occ.importShapes("part.step")
gmsh.model.occ.healShapes()
gmsh.model.occ.synchronize()

# Re-discover entities — don't rely on dimTags from importShapes
all_vols = gmsh.model.getEntities(3)
all_surfs = gmsh.model.getEntities(2)

# Use spatial queries to identify specific features
fixed_surfs = gmsh.model.getEntitiesInBoundingBox(
    -0.01, -0.01, -0.01, 10.01, 10.01, 0.01, dim=2
)  # surfaces near z=0
```

---

## The `synchronize()` Contract

### What it does

`synchronize()` transfers the geometry from the kernel's internal representation into the Gmsh model. Before synchronize, entities exist only inside the kernel. After, they are visible to the rest of Gmsh (entity queries, mesh generation, physical groups).

```python
gmsh.model.occ.synchronize()   # for OCC kernel
gmsh.model.geo.synchronize()   # for built-in kernel
```

### When you must call it

**Before** any of these:
- `gmsh.model.getEntities()`, `getBoundary()`, `getAdjacencies()`, or any entity query
- `gmsh.model.addPhysicalGroup()`
- `gmsh.model.mesh.generate()`
- `gmsh.model.mesh.setSize()`, `setTransfiniteCurve()`, or any mesh control

**After** any geometry creation or modification:
- Adding points, curves, surfaces, volumes
- Boolean operations
- Transformations
- Extrusion / revolution
- CAD import
- Healing

### Can you call it multiple times?

Yes. It's safe and idempotent for geometry that hasn't changed. The typical pattern is to synchronize after each logical block of geometry operations:

```python
# Block 1: create primitives
box = gmsh.model.occ.addBox(...)
cyl = gmsh.model.occ.addCylinder(...)
gmsh.model.occ.synchronize()     # entities now queryable

# Block 2: boolean
out, out_map = gmsh.model.occ.fragment(...)
gmsh.model.occ.synchronize()     # new entities now queryable

# Block 3: assign physical groups (requires sync'd entities)
gmsh.model.addPhysicalGroup(...)
```

### What breaks without it

Without `synchronize()`:
- Entity queries return empty or stale results
- Physical group assignment fails (entities don't "exist" yet)
- Mesh generation has nothing to mesh
- Bounding box queries return wrong values

The error is usually silent — you get empty results, not an exception. This makes it the most common source of "it runs but nothing happens" bugs.

### Cost

`synchronize()` rebuilds the BRep topology, which has a cost proportional to model complexity. For small models it's negligible. For large models with many boolean operations, batching geometry changes and synchronizing once at the end is more efficient than synchronizing after every call.

---

## Summary — Tag Flow Through Operations

| Operation          | Input tags       | Output tags           | Tracking method                    |
| ------------------ | ---------------- | --------------------- | ---------------------------------- |
| Bottom-up creation | —                | returned directly     | Capture return value               |
| OCC primitives     | —                | top-level returned    | Return value + `getBoundary()`     |
| Extrusion/revolve  | preserved        | returned list         | Index into return list             |
| Transforms         | unchanged        | same as input         | No tracking needed                 |
| Copy               | unchanged        | new tags returned     | Positional mapping                 |
| Booleans           | **destroyed**    | new tags              | `outDimTagsMap` (parallel to input)|
| CAD import         | —                | returned list         | Capture return + entity queries    |
| Heal               | **may change**   | unpredictable         | Re-query with `getEntities()`      |
| Remove             | **destroyed**    | —                     | Tags freed                         |
