# Gmsh Selection and Querying

How to find, filter, and retrieve BRep entities and physical groups — both by direct handle and by geometric criteria.

## Two Namespaces, Two Selection Targets

Gmsh maintains two independent `(dim, tag)` namespaces (see [[gmsh_basics]]):

**BRep entities** — the geometric model: points, curves, surfaces, volumes. These are the atoms that Gmsh meshes. You select them by tag, by adjacency, by bounding box, or by geometric properties.

**Physical groups** — semantic labels layered on top of BRep entities. A physical group collects entity tags of a single dimension under a name. You select physical groups by name or tag, then resolve them to the underlying entities.

Most selection workflows go: **find entities → assign to physical group → extract mesh data for that group**. The selection step is where the work is.

## Selection by Handle

The simplest selection: you already know the `(dim, tag)`.

### Direct Entity Enumeration

```python
# All entities in the model (or filter by dim)
all_entities = gmsh.model.getEntities(dim=-1)        # all dims
all_surfaces = gmsh.model.getEntities(dim=2)          # surfaces only
all_curves   = gmsh.model.getEntities(dim=1)          # curves only
```

`getEntities` returns a list of `(dim, tag)` pairs. With `dim=-1` (default), it returns everything across all dimensions. This is the **universe** — the full set you filter down from.

There is also a pre-synchronize variant in the OCC kernel:

```python
# Entities in the OCC kernel (before synchronize)
occ_entities = gmsh.model.occ.getEntities(dim=-1)
```

The distinction matters: `gmsh.model.getEntities` reads the **synced BRep model**. `gmsh.model.occ.getEntities` reads the **OCC kernel state**, which may contain entities not yet synced. After `synchronize()`, both return the same result.

### Entity Properties

Once you have a `(dim, tag)`, you can query its properties:

```python
# Type — what kind of geometric entity
entity_type = gmsh.model.getType(dim, tag)
# Returns strings like "Line", "Circle", "Plane", "BSplineSurface", etc.

# Properties — geometric coefficients
integers, reals = gmsh.model.getEntityProperties(dim, tag)
# For a plane surface: reals = [a, b, c, d] (ax + by + cz + d = 0)
# For a sphere: reals = [cx, cy, cz, r]
# For a cylinder: reals = [cx, cy, cz, ax, ay, az, r]

# Name — optional user-assigned string
name = gmsh.model.getEntityName(dim, tag)
```

### Physical Group Lookup

Physical groups are the bridge between geometry and solver semantics. The API provides bidirectional lookup:

```python
# ─── From physical group → entities ───

# All physical groups (or filter by dim)
pgs = gmsh.model.getPhysicalGroups(dim=-1)     # list of (dim, pg_tag)

# Entity tags in a physical group (by dim + pg_tag)
entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim=2, tag=pg_tag)
# Returns: [5, 8, 12] — entity tags (not dimtags), all at the given dim

# Entity dimtags by physical group name
dimtags = gmsh.model.getEntitiesForPhysicalName("supports")
# Returns: [(0, 1), (0, 5), (0, 9)] — full (dim, tag) pairs
# Note: if multiple PGs share the name, returns the union

# Batch: all groups AND their entities at once
pg_dimtags, entities_per_pg = gmsh.model.getPhysicalGroupsEntities(dim=-1)
# pg_dimtags[i] = (dim, pg_tag)
# entities_per_pg[i] = [(dim, entity_tag), ...] for that group

# ─── From entity → physical groups ───

# Which physical groups contain this entity?
pg_tags = gmsh.model.getPhysicalGroupsForEntity(dim=2, tag=5)
# Returns: [1, 3] — physical group tags (at dim=2)
```

The key asymmetry: `getEntitiesForPhysicalGroup` takes `dim` + `tag` and returns **bare tags** (because all entities in a PG share the same dim). `getEntitiesForPhysicalName` returns **dimtags** (because the name might match PGs at different dims, though typically it doesn't).

### Physical Group Name ↔ Tag

```python
# Get the name of a physical group
name = gmsh.model.getPhysicalName(dim=2, tag=1)

# Set / rename
gmsh.model.setPhysicalName(dim=2, tag=1, name="floor_slab")
```

Remember from [[gmsh_basics]]: physical group names can be duplicated across dimensions. A physical group named `"boundary"` at dim=1 and another named `"boundary"` at dim=2 are distinct groups.

## Selection by Bounding Box

When you don't know the tags — common after CAD import, boolean operations, or fragmentation — you select entities by their spatial location.

### getEntitiesInBoundingBox

The primary spatial query. Returns all entities whose **own bounding box** intersects the query box:

```python
dimtags = gmsh.model.getEntitiesInBoundingBox(
    xmin, ymin, zmin,    # lower corner
    xmax, ymax, zmax,    # upper corner
    dim=-1,              # restrict to this dim, -1 = all
)
```

```
   Parameters
   ──────────────────────────────
   xmin, ymin, zmin : float — lower corner of query box (coordinates, not tags)
   xmax, ymax, zmax : float — upper corner of query box (coordinates, not tags)
   dim              : int   — dimension filter (-1 = all)

   Returns
   ──────────────────────────────
   list[(dim, tag)] — entities whose bounding box overlaps the query box
```

The test is **bounding box overlap**, not containment. An entity whose bounding box extends outside the query region will still be returned if any part of its bbox intersects.

Example — find all surfaces near the base of a structure:

```python
# Select surfaces near z = 0 (within ±0.1 tolerance)
base_surfaces = gmsh.model.getEntitiesInBoundingBox(
    -1e10, -1e10, -0.1,   # xmin, ymin, zmin
     1e10,  1e10,  0.1,   # xmax, ymax, zmax
    dim=2,
)
```

The large x/y bounds act as "don't care" — we only constrain z. This is a common pattern for plane selection.

There is also a pre-synchronize OCC variant:

```python
# Same query, but on OCC kernel state (before synchronize)
dimtags = gmsh.model.occ.getEntitiesInBoundingBox(
    xmin, ymin, zmin, xmax, ymax, zmax, dim=-1,
)
```

### getBoundingBox

Query the bounding box **of** an entity (the reverse direction — from entity to box):

```python
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
```

Special case — bounding box of the **entire model**:

```python
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
```

Passing `dim=-1, tag=-1` returns the global bounding box. This is useful for setting up field sizes or camera views.

Again, the OCC variant exists:

```python
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
```

### Composing Bounding Box Queries

A common workflow: use one entity's bounding box to find related entities:

```python
# Find all points on the boundary of surface 5
x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, 5)

# Expand slightly for tolerance
eps = 0.01
points_on_surface = gmsh.model.getEntitiesInBoundingBox(
    x0 - eps, y0 - eps, z0 - eps,
    x1 + eps, y1 + eps, z1 + eps,
    dim=0,
)
```

The `eps` expansion is critical. Bounding boxes are axis-aligned and tight — entities exactly on the boundary may fail the overlap test due to floating-point precision. Always pad.

## Selection by Topology

Topology queries navigate the BRep hierarchy — up, down, and sideways.

### getBoundary

Returns entities one dimension lower that form the boundary:

```python
boundary_dimtags = gmsh.model.getBoundary(
    dimTags,                  # input entities as [(dim, tag), ...]
    combined=True,            # True → boundary of the union
    oriented=False,           # True → tags carry sign (orientation)
    recursive=False,          # True → recurse down to dim 0
)
```

The `combined` flag controls whether you get the boundary of each entity separately (duplicates at shared interfaces) or the topological boundary of the union (shared faces cancel out):

```python
# Two adjacent volumes sharing face 10
vols = [(3, 1), (3, 2)]

# combined=True → face 10 cancels, only external faces remain
external = gmsh.model.getBoundary(vols, combined=True)

# combined=False → face 10 appears twice (once per volume)
all_faces = gmsh.model.getBoundary(vols, combined=False)
```

With `recursive=True`, the query walks all the way down to points:

```python
# Get all points on volume 1
pts = gmsh.model.getBoundary([(3, 1)], recursive=True)
# Returns dim-0 entities — every vertex of every face of the volume
```

The `oriented` flag adds sign to tags: a positive tag means the boundary entity's orientation agrees with the parent's induced orientation; negative means it's reversed. This matters for curve loops and surface loops.

### getAdjacencies

Navigates the BRep tree in both directions from a single entity:

```python
upward, downward = gmsh.model.getAdjacencies(dim, tag)
# upward   → tags at dim+1 that contain this entity
# downward → tags at dim-1 on this entity's boundary
```

```
   Entity (1, 5) — a curve
   ├── upward   = [2, 7]     — surfaces 2 and 7 use this curve
   └── downward = [3, 8]     — points 3 and 8 are the curve's endpoints
```

Note: `getAdjacencies` returns **tags**, not dimtags. The dimension is implied: upward is always `dim+1`, downward is always `dim-1`.

### isEntityOrphan

Checks whether an entity is disconnected from the highest-dimension entities in the model:

```python
is_orphan = gmsh.model.isEntityOrphan(dim, tag)
# Returns 1 if orphan, 0 if connected
```

An orphan point or curve floats free — it's not part of any surface or volume boundary. This commonly happens after boolean operations that split entities, leaving fragments that aren't connected to the main topology.

## Selection by Geometric Properties

Beyond bounding boxes, Gmsh provides richer geometric queries.

### Closest Point / Closest Entity

Project a point onto an entity (curve or surface):

```python
# Closest point on curve 3 to position (5.0, 2.0, 0.0)
closest_xyz, param_coord = gmsh.model.getClosestPoint(
    dim=1, tag=3,
    coord=[5.0, 2.0, 0.0],
)
# closest_xyz    = [5.0, 1.8, 0.0] — actual closest point on the curve
# param_coord    = [0.73]           — parametric coordinate on the curve
```

For finding the closest entity from a set (OCC kernel only):

```python
# Which of these curves is closest to point (5, 2, 0)?
candidates = gmsh.model.occ.getEntities(dim=1)
out_dimtags, distances, coords = gmsh.model.occ.getClosestEntities(
    x=5.0, y=2.0, z=0.0,
    dimTags=candidates,
    n=3,  # return top 3 closest
)
# out_dimtags = [(1, 5), (1, 12), (1, 3)] — sorted by distance
# distances   = [0.2, 0.8, 1.1]
# coords      = [x1,y1,z1, x2,y2,z2, x3,y3,z3] — closest points
```

### Point Inside Test

Check whether coordinates fall inside an entity:

```python
n_inside = gmsh.model.isInside(
    dim=2, tag=5,
    coord=[1.0, 2.0, 0.0,   3.0, 4.0, 0.0],  # two test points, concatenated
    parametric=False,
)
# Returns: number of points that are inside entity (2, 5)
```

This is only available for certain entity types (planes, common surfaces). Not universally supported.

### Distance Between Entities (OCC)

```python
distance, x1, y1, z1, x2, y2, z2 = gmsh.model.occ.getDistance(
    dim1=1, tag1=3,    # first entity
    dim2=1, tag2=7,    # second entity
)
# distance = minimum distance between the two entities
# (x1,y1,z1) and (x2,y2,z2) = the closest points on each entity
```

### Mass (Length / Area / Volume)

```python
mass = gmsh.model.occ.getMass(dim, tag)
# dim=1 → curve length
# dim=2 → surface area
# dim=3 → volume
```

This is useful for filtering: "all curves longer than 5 meters" or "all surfaces with area > 10 m²".

### Parametric Evaluation

Every curve and surface in Gmsh has a parametric representation. You can evaluate it:

```python
# Evaluate curve 3 at parametric coordinate t = 0.5
xyz = gmsh.model.getValue(dim=1, tag=3, parametricCoord=[0.5])
# Returns: [x, y, z] — the point on the curve at t=0.5

# Get parametric bounds
bounds_min, bounds_max = gmsh.model.getParametrizationBounds(dim=1, tag=3)
# bounds_min = [0.0], bounds_max = [1.0] — typical for curves

# Reverse: from (x,y,z) → parametric coordinate
param = gmsh.model.getParametrization(dim=1, tag=3, coord=[5.0, 2.0, 0.0])
# Returns: [0.73] — the parametric coordinate nearest to that point
```

For surfaces, parametric coordinates are `(u, v)` pairs:

```python
# Evaluate surface 5 at (u=0.3, v=0.7)
xyz = gmsh.model.getValue(dim=2, tag=5, parametricCoord=[0.3, 0.7])

# Multiple points at once — concatenated
xyzs = gmsh.model.getValue(dim=2, tag=5,
    parametricCoord=[0.0, 0.0,  0.5, 0.5,  1.0, 1.0])
# Returns: [x1,y1,z1, x2,y2,z2, x3,y3,z3]
```

### Curvature and Normals

```python
# Curvature of curve 3 at t = 0.5
curvatures = gmsh.model.getCurvature(dim=1, tag=3, parametricCoord=[0.5])

# Surface normal at (u, v) = (0.3, 0.7)
normals = gmsh.model.getNormal(tag=5, parametricCoord=[0.3, 0.7])
# Returns: [nx, ny, nz]

# Principal curvatures of a surface
kmax, kmin, dmax, dmin = gmsh.model.getPrincipalCurvatures(
    tag=5, parametricCoord=[0.3, 0.7],
)
```

## Combining Selection Methods

Real selection workflows combine multiple methods. Here's the general pattern:

### Pattern: Plane Selection via Bounding Box

Select all entities on a specific plane (e.g., z = 0):

```python
tol = 1e-3
base_points = gmsh.model.getEntitiesInBoundingBox(
    -1e10, -1e10, -tol,
     1e10,  1e10,  tol,
    dim=0,
)
gmsh.model.addPhysicalGroup(0, [t for _, t in base_points], name="fixed_supports")
```

### Pattern: Region Selection After Fragment

After `fragment()`, original tags are gone. Find entities by location:

```python
# Fragment produced conformal mesh at the interface
gmsh.model.occ.fragment(objects, tools)
gmsh.model.occ.synchronize()

# Find the left column (entities around x=0, y=0)
eps = 0.5
left_col_curves = gmsh.model.getEntitiesInBoundingBox(
    -eps, -eps, -1e10,
     eps,  eps,  1e10,
    dim=1,
)
```

### Pattern: Topology Walk

Find all surfaces bounding a volume, then all curves on those surfaces:

```python
# Surfaces of volume 1
faces = gmsh.model.getBoundary([(3, 1)], combined=False, oriented=False)

# Curves on those surfaces
edges = gmsh.model.getBoundary(faces, combined=False, oriented=False)

# Or in one step with recursive
all_points = gmsh.model.getBoundary([(3, 1)], recursive=True)
```

### Pattern: Filter by Orientation

Select vertical curves (aligned with z-axis) using parametric evaluation:

```python
import numpy as np

vertical_curves = []
for dim, tag in gmsh.model.getEntities(dim=1):
    bounds = gmsh.model.getParametrizationBounds(1, tag)
    t0, t1 = bounds[0][0], bounds[1][0]
    p0 = np.array(gmsh.model.getValue(1, tag, [t0]))
    p1 = np.array(gmsh.model.getValue(1, tag, [t1]))
    direction = p1 - p0
    direction /= np.linalg.norm(direction) + 1e-30
    if abs(abs(direction[2]) - 1.0) < 0.01:  # aligned with z
        vertical_curves.append((1, tag))
```

### Pattern: Entity Properties

Select all planar surfaces (useful for slab identification):

```python
planes = []
for dim, tag in gmsh.model.getEntities(dim=2):
    etype = gmsh.model.getType(dim, tag)
    if etype == "Plane":
        planes.append((dim, tag))
```

## Pre-sync vs Post-sync Queries

A critical distinction:

| Function | Namespace | Requires `synchronize()` |
| --- | --- | --- |
| `gmsh.model.getEntities` | Synced BRep model | Yes |
| `gmsh.model.getEntitiesInBoundingBox` | Synced BRep model | Yes |
| `gmsh.model.getBoundingBox` | Synced BRep model | Yes |
| `gmsh.model.occ.getEntities` | OCC kernel | No |
| `gmsh.model.occ.getEntitiesInBoundingBox` | OCC kernel | No |
| `gmsh.model.occ.getBoundingBox` | OCC kernel | No |

The OCC variants are useful when you need to query geometry **during construction** — between OCC calls but before `synchronize()`. The model variants are the standard post-sync queries.

After `synchronize()`, both return the same results. Prefer the `gmsh.model` variants in general — they work regardless of which kernel was used.

## Summary of Selection API

| Method | Input | Output | Use case |
| --- | --- | --- | --- |
| `getEntities(dim)` | Dimension filter | `[(dim, tag), ...]` | Universe enumeration |
| `getEntitiesInBoundingBox(...)` | 6 coords + dim | `[(dim, tag), ...]` | Spatial selection |
| `getBoundingBox(dim, tag)` | Entity handle | 6 floats | Entity extent |
| `getBoundary(dimTags, ...)` | Entity list | `[(dim, tag), ...]` | Topology walk (down) |
| `getAdjacencies(dim, tag)` | Entity handle | (up_tags, down_tags) | Topology walk (both) |
| `getClosestPoint(dim, tag, coord)` | Entity + point | xyz + param | Projection |
| `occ.getClosestEntities(x,y,z, ...)` | Point + candidates | Sorted entities | Proximity search |
| `occ.getDistance(d1,t1, d2,t2)` | Two entities | Distance + points | Inter-entity distance |
| `occ.getMass(dim, tag)` | Entity handle | Float | Length/area/volume |
| `isInside(dim, tag, coord)` | Entity + points | Count | Containment test |
| `getType(dim, tag)` | Entity handle | String | Geometric classification |
| `getEntitiesForPhysicalGroup(dim, tag)` | PG handle | `[tag, ...]` | PG → entities |
| `getEntitiesForPhysicalName(name)` | PG name | `[(dim, tag), ...]` | PG name → entities |
| `getPhysicalGroupsForEntity(dim, tag)` | Entity handle | `[pg_tag, ...]` | Entity → PGs |

## Cross-Reference

- [[gmsh_basics]] — BRep model, (dim,tag) identity, physical groups
- [[gmsh_geometry_basics]] — Tag tracking through geometry operations
- [[gmsh_interface]] — API design, elementary types contract
