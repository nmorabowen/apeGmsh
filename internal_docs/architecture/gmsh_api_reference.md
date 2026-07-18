# Gmsh API Reference

Two compilations: the **essential calls** for the apeGmsh→OpenSees structural workflow, and the **deep-dive functions** for when you need full control.

---

# Part I — Workflow Essentials

The 25 calls that carry 90% of a structural analysis pipeline. Organized by workflow stage.

## Stage 1: Session

```python
gmsh.initialize()                    # First call — always
gmsh.model.add("my_model")           # Create a named model
# ... everything happens here ...
gmsh.finalize()                      # Last call — always
```

Nothing else works without `initialize()`. One session can hold multiple models; `add()` creates and sets the current one.

## Stage 2: Geometry

### Points and Lines

```python
p1 = gmsh.model.occ.addPoint(x, y, z)          # → tag (int)
p2 = gmsh.model.occ.addPoint(x, y, z, meshSize) # optional size hint at point

l1 = gmsh.model.occ.addLine(p1, p2)             # → tag (int)
```

### Curves → Surfaces → Volumes (bottom-up)

```python
cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])   # → curve loop tag
s  = gmsh.model.occ.addPlaneSurface([cl])              # → surface tag
sl = gmsh.model.occ.addSurfaceLoop([s1, s2, ...])     # → surface loop tag
v  = gmsh.model.occ.addVolume([sl])                    # → volume tag
```

### OCC Primitives (high-level shortcuts)

```python
box = gmsh.model.occ.addBox(x, y, z, dx, dy, dz)          # → volume tag
cyl = gmsh.model.occ.addCylinder(x, y, z, dx, dy, dz, r)  # → volume tag
sph = gmsh.model.occ.addSphere(cx, cy, cz, r)             # → volume tag
```

### Transforms

```python
gmsh.model.occ.translate(dimTags, dx, dy, dz)
gmsh.model.occ.rotate(dimTags, cx, cy, cz, ax, ay, az, angle)  # angle in radians
```

### Extrude

```python
result = gmsh.model.occ.extrude(dimTags, dx, dy, dz)
# result = [(top_dim, top_tag), (swept_dim, swept_tag), (lateral...), ...]
```

### Fragment (conformal interfaces)

```python
out_dimtags, result_map = gmsh.model.occ.fragment(
    objectDimTags, toolDimTags,
    removeObject=True, removeTool=True,
)
# result_map[i] = list of new dimtags that replaced input[i]
```

This is the single most important boolean operation for structural analysis. It makes the mesh conformal at material interfaces.

### Synchronize

```python
gmsh.model.occ.synchronize()    # MANDATORY after any occ.* geometry call
```

Nothing downstream (queries, meshing, physical groups) sees OCC changes until you synchronize. Call it after geometry construction is complete, or after each batch of operations.

## Stage 3: Physical Groups

```python
# Create
pg_tag = gmsh.model.addPhysicalGroup(dim, [tag1, tag2, ...], name="columns")

# Query
pgs            = gmsh.model.getPhysicalGroups(dim=-1)          # all PGs
entity_tags    = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)  # PG → entity tags
name           = gmsh.model.getPhysicalName(dim, pg_tag)       # PG tag → name
```

Physical groups are how the solver knows which entities are columns, which are slabs, where the supports are, and what material goes where. Every entity that matters to the solver must live in a physical group.

## Stage 4: Mesh

### Configuration

```python
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay
```

### Local sizing

```python
# Size at specific points
gmsh.model.mesh.setSize([(0, p1), (0, p2)], size=0.1)
```

### Generate

```python
gmsh.model.mesh.generate(3)    # 1D → 2D → 3D in one call
```

### Extract

```python
# All nodes
nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
# nodeCoords is flat: [x1,y1,z1, x2,y2,z2, ...] — reshape to (-1, 3)

# All elements (per dimension)
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)
# elemTypes[i]    = element type code
# elemTags[i]     = array of element tags for that type
# elemNodeTags[i] = flat connectivity array for that type

# Element type metadata
name, dim, order, npe, ref_coords, n_prim = gmsh.model.mesh.getElementProperties(elemType)
# npe = nodes per element — use to reshape connectivity

# Nodes by physical group
nodeTags, nodeCoords = gmsh.model.mesh.getNodesForPhysicalGroup(dim, pg_tag)
```

### Export

```python
gmsh.write("model.msh")     # MSH format (default v4)
```

## Stage 5: Entity Queries

```python
# All entities
all_ents = gmsh.model.getEntities(dim=-1)       # [(dim, tag), ...]
surfaces = gmsh.model.getEntities(dim=2)         # surfaces only

# Bounding box query
found = gmsh.model.getEntitiesInBoundingBox(
    xmin, ymin, zmin, xmax, ymax, zmax, dim=0,
)

# Entity bounding box
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

# Boundary walk
boundary_dimtags = gmsh.model.getBoundary(
    [(3, vol_tag)], combined=True, oriented=False,
)
```

## The Complete Minimal Pipeline

Putting it all together — the skeleton of every apeGmsh model:

```python
import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("structure")

# ── Geometry ──
p1 = gmsh.model.occ.addPoint(0, 0, 0)
p2 = gmsh.model.occ.addPoint(10, 0, 0)
p3 = gmsh.model.occ.addPoint(10, 0, 5)
p4 = gmsh.model.occ.addPoint(0, 0, 5)
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)
gmsh.model.occ.synchronize()

# ── Physical groups ──
gmsh.model.addPhysicalGroup(1, [l1], name="beam")
gmsh.model.addPhysicalGroup(1, [l2, l4], name="columns")
gmsh.model.addPhysicalGroup(0, [p1, p4], name="fixed")

# ── Mesh ──
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
gmsh.model.mesh.generate(1)

# ── Extract ──
for dim, pg_tag in gmsh.model.getPhysicalGroups():
    name = gmsh.model.getPhysicalName(dim, pg_tag)
    tags = gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag)
    # → send to OpenSees ...

nodeTags, coords, _ = gmsh.model.mesh.getNodes()
coords = np.reshape(coords, (-1, 3))

gmsh.finalize()
```

---

# Part II — Deep Dive

Functions that matter when you need precision, when defaults break, or when you're pushing Gmsh to its limits. These are the calls that separate someone who uses Gmsh from someone who understands it.

## Geometric Intelligence

### Parametric Evaluation

Every curve and surface in Gmsh has a parametric representation. These calls let you walk along geometry programmatically.

```python
# Get parametric bounds of a curve
t_min, t_max = gmsh.model.getParametrizationBounds(dim=1, tag=curve_tag)
# t_min = [0.0], t_max = [1.0] — typically

# Evaluate: parametric → physical coordinates
xyz = gmsh.model.getValue(dim=1, tag=curve_tag, parametricCoord=[0.5])
# Returns: [x, y, z] — the midpoint of the curve

# Reverse: physical → parametric
param = gmsh.model.getParametrization(dim=1, tag=curve_tag, coord=[x, y, z])
# Returns: [t] — the parametric coordinate nearest to that point
```

**Why this matters.** Orientation checks. Midpoint extraction. Custom mesh grading based on curve position. Sampling geometry for verification without meshing.

For surfaces, parametric coordinates are `(u, v)` pairs:

```python
# Surface normal at (u, v) = (0.3, 0.7)
normals = gmsh.model.getNormal(tag=surf_tag, parametricCoord=[0.3, 0.7])
# Returns: [nx, ny, nz]

# Derivative at a point — tangent vectors
derivatives = gmsh.model.getDerivative(dim=2, tag=surf_tag,
    parametricCoord=[0.3, 0.7])
# Returns: [du_x, du_y, du_z, dv_x, dv_y, dv_z]
```

### Curvature

```python
# Curve curvature at parametric point
kappa = gmsh.model.getCurvature(dim=1, tag=curve_tag, parametricCoord=[0.5])

# Surface principal curvatures
kmax, kmin, dir_max, dir_min = gmsh.model.getPrincipalCurvatures(
    tag=surf_tag, parametricCoord=[0.3, 0.7],
)
```

**Why this matters.** Curvature-based mesh refinement. If you're writing a custom size callback, you need curvature to decide where the mesh needs to be finer.

### Closest Point Projection

```python
# Project a point onto a curve
closest_xyz, param = gmsh.model.getClosestPoint(
    dim=1, tag=curve_tag, coord=[x, y, z],
)
# closest_xyz = nearest point on the curve
# param       = parametric coordinate of that point
```

**Why this matters.** Node snapping. Force application at specific geometry locations. Verifying that mesh nodes land where they should.

### OCC Proximity and Distance

```python
# Closest entities from a set
out_dimtags, distances, coords = gmsh.model.occ.getClosestEntities(
    x, y, z,
    dimTags=candidates,
    n=3,                   # top 3 closest
)

# Minimum distance between two entities
dist, x1, y1, z1, x2, y2, z2 = gmsh.model.occ.getDistance(
    dim1, tag1, dim2, tag2,
)
```

**Why this matters.** Automated assignment of loads to nearest geometry. Clearance checks between structural members. Quality assurance on imported CAD.

### Point-in-Entity Test

```python
n_inside = gmsh.model.isInside(dim=2, tag=surf_tag,
    coord=[x1,y1,z1, x2,y2,z2],   # multiple points, concatenated
    parametric=False,
)
```

Limited to certain entity types, but useful when available.

### Entity Type and Properties

```python
# What kind of geometry is this?
etype = gmsh.model.getType(dim, tag)
# "Line", "Circle", "BSplineCurve", "Plane", "Cylinder", "BSplineSurface", ...

# Geometric coefficients
integers, reals = gmsh.model.getEntityProperties(dim, tag)
# Plane:    reals = [a, b, c, d]  (ax + by + cz + d = 0)
# Sphere:   reals = [cx, cy, cz, r]
# Cylinder: reals = [cx, cy, cz, ax, ay, az, r]
# Torus:    reals = [cx, cy, cz, ax, ay, az, R, r]
```

**Why this matters.** Programmatic selection: "find all planar surfaces" or "find all circular curves". After CAD import, entity types tell you what you're working with without visual inspection.

### Mass (Length / Area / Volume)

```python
mass = gmsh.model.occ.getMass(dim, tag)
# dim=1 → length, dim=2 → area, dim=3 → volume
```

**Why this matters.** Validation: "does this column have the expected length?" Filtering: "select all curves longer than 3 meters." Weight calculation for gravity loads.

### Center of Mass

```python
cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
```

Heavily used in apeGmsh — over 50 call sites. This is the main way to locate entities spatially for sorting, classification, and spatial filters.

## Topology Navigation

### Adjacency — the BRep Graph

```python
upward, downward = gmsh.model.getAdjacencies(dim, tag)
# upward   = tags at dim+1 containing this entity
# downward = tags at dim-1 bounding this entity
```

**Why this matters.** "Which volumes share this surface?" → interface detection for multi-material models. "Which curves bound this surface?" → structured mesh setup requires knowing corner points.

### Boundary with Recursion

```python
# Direct boundary (one level down)
faces = gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False)

# Recursive to points
all_pts = gmsh.model.getBoundary([(3, vol)], recursive=True)

# Combined boundary — shared faces cancel
external = gmsh.model.getBoundary(vol_list, combined=True)
```

The `combined=True` trick: if two volumes share a face, that face does not appear in the combined boundary. This gives you the **external skin** of a multi-volume model — exactly what you need for applying surface loads or support conditions.

### Orphan Detection

```python
is_orphan = gmsh.model.isEntityOrphan(dim, tag)
```

After fragment, you may have floating points or curves that are not connected to the main topology. These create orphan nodes in the mesh. Detect and handle before meshing.

### Curve Loop and Surface Loop Decomposition (OCC)

```python
# Which curves make up surface 5?
loop_tags, curve_tags_per_loop = gmsh.model.occ.getCurveLoops(surf_tag)

# Which surfaces make up volume 1?
sloop_tags, surf_tags_per_sloop = gmsh.model.occ.getSurfaceLoops(vol_tag)
```

**Why this matters.** Transfinite meshing requires you to know the exact curves bounding a surface and the exact surfaces bounding a volume. These calls give you that decomposition without guessing.

## Advanced Mesh Control

### Size Callback

```python
def my_size(dim, tag, x, y, z, lc):
    # lc = the "default" size Gmsh computed
    if z > 10.0:
        return 0.1   # fine mesh above z=10
    return lc        # default elsewhere

gmsh.model.mesh.setSizeCallback(my_size)
```

**Why this matters.** Fields are powerful but sometimes you need logic that depends on the entity metadata, physical group membership, or spatial criteria that fields can't express. The callback is the escape hatch.

**Caveat:** Called per node evaluation during meshing — must be fast. Avoid API calls inside the callback (use precomputed lookups).

### Mesh Fields — The Power Tools

```python
# Distance field — distance from a set of curves
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [1, 2, 3])
gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

# Threshold — map distance to element size
f_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", 0.05)    # near
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", 0.5)     # far
gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)     # start grading at
gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 2.0)     # full coarse at

# BoundaryLayer — anisotropic refinement near surfaces
f_bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(f_bl, "CurvesList", [5, 6])
gmsh.model.mesh.field.setNumber(f_bl, "Size", 0.01)           # first layer
gmsh.model.mesh.field.setNumber(f_bl, "Ratio", 1.3)           # growth ratio
gmsh.model.mesh.field.setNumber(f_bl, "Quads", 1)             # quad layers
gmsh.model.mesh.field.setAsBoundaryLayer(f_bl)

# MathEval — size as a mathematical expression
f_math = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f_math, "F", "0.1 + 0.01*z")

# Box — uniform size in a box region
f_box = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(f_box, "VIn", 0.05)     # inside size
gmsh.model.mesh.field.setNumber(f_box, "VOut", 1.0)     # outside size
gmsh.model.mesh.field.setNumber(f_box, "XMin", 0)
gmsh.model.mesh.field.setNumber(f_box, "XMax", 10)
gmsh.model.mesh.field.setNumber(f_box, "YMin", 0)
gmsh.model.mesh.field.setNumber(f_box, "YMax", 5)
gmsh.model.mesh.field.setNumber(f_box, "ZMin", 0)
gmsh.model.mesh.field.setNumber(f_box, "ZMax", 3)

# Min — combine fields (smallest size wins)
f_min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thresh, f_box])
gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
```

**The composition pattern:** Distance → Threshold → Min. Multiple regions of refinement, each defined by distance to a geometric feature, combined with Min so the finest controls.

### Transfinite (Structured) Meshing

```python
# Structured curve: exactly N nodes, with geometric progression
gmsh.model.mesh.setTransfiniteCurve(curve_tag, numNodes=20, meshType="Progression", coef=1.1)

# Structured surface: quad mesh (requires 3 or 4 boundary curves)
gmsh.model.mesh.setTransfiniteSurface(surf_tag, arrangement="Left",
    cornerTags=[p1, p2, p3, p4])

# Structured volume: hex mesh (requires 6 boundary surfaces)
gmsh.model.mesh.setTransfiniteVolume(vol_tag,
    cornerTags=[p1, p2, p3, p4, p5, p6, p7, p8])

# Recombine triangles → quads on a surface
gmsh.model.mesh.setRecombine(dim=2, tag=surf_tag)
```

**Why this matters.** Structural analysis often needs hex or quad elements for accuracy. Transfinite meshing is the only way to get fully structured hex meshes in Gmsh. But it's fragile — wrong corner ordering or incompatible curve node counts will fail silently.

### Mesh Element Order

```python
gmsh.model.mesh.setOrder(2)    # Convert to quadratic elements
```

Called after `generate()`. Converts all linear elements to quadratic by adding mid-edge nodes. For structural analysis, quadratic elements typically outperform linear ones at the same mesh density.

### Node Deduplication

```python
gmsh.model.mesh.removeDuplicateNodes()
```

After importing or merging meshes, duplicate nodes at shared interfaces may exist. This fuses them (within tolerance). Different from `fragment()` — this operates on the mesh, not the geometry.

## CAD Import and Repair

### Import

```python
out_dimtags = gmsh.model.occ.importShapes("structure.step",
    highestDimOnly=True)
# highestDimOnly=True → skip free curves/points from CAD
# Returns: list of (dim, tag) for imported entities
```

Also supports IGES and BREP formats. After import, always synchronize:

```python
gmsh.model.occ.synchronize()
```

### Heal

```python
healed = gmsh.model.occ.healShapes(
    dimTags=[],             # empty = heal everything
    tolerance=1e-8,
    fixDegenerated=True,
    fixSmallEdges=True,
    fixSmallFaces=True,
    sewFaces=True,
    makeSolids=True,
)
```

**Why this matters.** STEP/IGES files from real CAD tools are never clean. Small edges, degenerate faces, unsealed solids. `healShapes` is OCC's repair toolkit. After healing, tags may change — the returned `healed` dimtags are the new entity identifiers.

### Remove Duplicates

```python
gmsh.model.occ.removeAllDuplicates()
```

Merges geometrically coincident entities across all dimensions. Essential after IGES import where shared vertices between beams are stored as separate points.

## Partitioning

```python
# Partition the mesh into N parts
gmsh.model.mesh.partition(numParts)

# Or with explicit element weights (for load balancing)
gmsh.model.mesh.partition(numParts, elementWeights=weights_array)

# Query partition info
n = gmsh.model.getNumberOfPartitions()
partitions = gmsh.model.getPartitions(dim, tag)    # which partitions own this entity
p_dim, p_tag = gmsh.model.getParent(dim, tag)      # parent entity before partition
```

**Partition options** (set before `partition()`):

```python
gmsh.option.setNumber("Mesh.PartitionCreateTopology", 1)   # create partition entities
gmsh.option.setNumber("Mesh.PartitionCreatePhysicals", 1)  # create per-partition PGs
gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles", 0)   # 0=single file, 1=per-partition
```

**Why this matters.** OpenSeesMP needs per-partition data. After partitioning, Gmsh creates new partition entities with parent tracking — you can extract nodes/elements per partition and identify interface nodes for parallel assembly.

## Options — The Ones That Matter

### Mesh Sizing

| Option | Default | What it does |
| --- | --- | --- |
| `Mesh.MeshSizeMax` | 1e22 | Global upper bound on element size |
| `Mesh.MeshSizeMin` | 0 | Global lower bound |
| `Mesh.MeshSizeFromPoints` | 1 | Use size hints from points |
| `Mesh.MeshSizeFromCurvature` | 0 | Adaptive sizing from curvature |
| `Mesh.MeshSizeExtendFromBoundary` | 1 | Propagate boundary sizes inward |
| `Mesh.CharacteristicLengthFactor` | 1.0 | Global size multiplier |

### Algorithm Selection

| Option | Values | Notes |
| --- | --- | --- |
| `Mesh.Algorithm` | 1,2,5,6,7,8,9,11 | 2D algorithm. 6 (Frontal-Delaunay) is default and robust |
| `Mesh.Algorithm3D` | 1,4,7,10 | 3D algorithm. 1 (Delaunay) is default |

### Output Control

| Option | Default | What it does |
| --- | --- | --- |
| `Mesh.SaveAll` | 0 | If 1, save elements not in any physical group |
| `Mesh.MshFileVersion` | 4.1 | MSH format version |
| `General.Terminal` | 1 | Print to terminal (0 = suppress) |

## Logging and Diagnostics

```python
# Capture Gmsh messages programmatically
gmsh.logger.start()
gmsh.model.mesh.generate(3)
log = gmsh.logger.get()          # list of strings
gmsh.logger.stop()

# Timing
t0 = gmsh.logger.getWallTime()
# ... expensive operation ...
dt = gmsh.logger.getWallTime() - t0

# Memory
mem_mb = gmsh.logger.getMemory()
```

**Why this matters.** When meshing fails or produces poor quality, the log tells you why. Capture it programmatically instead of reading terminal output.

## Cross-Reference

- [[gmsh_basics]] — BRep model, session lifecycle, physical groups
- [[gmsh_geometry_basics]] — Geometry construction and tag tracking
- [[gmsh_meshing_basics]] — Mesh size hierarchy, meshing workflow
- [[gmsh_meshing_advanced]] — Algorithms, fields, optimization, embedded entities
- [[gmsh_partitioning]] — METIS, partition entities, OpenSeesMP pipeline
- [[gmsh_selection]] — Selection methods and querying
- [[gmsh_interface]] — API architecture, Python binding, GUI
