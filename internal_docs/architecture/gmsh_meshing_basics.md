# Gmsh Meshing Basics

How geometry becomes a mesh, what Gmsh assumes by default, and how the output is structured.

---

## BRep–Mesh Duality

The BRep defines continuous geometry. The mesh is a discrete approximation **classified onto** that geometry. This classification is the fundamental link — every mesh entity (node, element) knows which BRep entity it belongs to.

```
GEOMETRY (continuous)              MESH (discrete)
───────────────────                ─────────────────
BRep entities                      Nodes + Elements
identified by (dim, tag)           classified onto entities
```

Classification means:

- A node on a curve belongs to that curve entity, not to the surfaces on either side of it.
- A node at the intersection of three curves belongs to the point entity at that junction.
- An element on a surface belongs to that surface entity.

This is how mesh conformity works across entities. Two surfaces sharing an edge don't each get their own nodes along that edge — the nodes belong to the edge entity (dim 1) and are shared by both surface meshes.

```
Surface A (2, 1)          Surface B (2, 2)
┌───────────┬───────────┐
│  ·  ·  ·  │  ·  ·  ·  │
│  ·  ·  · ─┤─ ·  ·  ·  │
│  ·  ·  ·  │  ·  ·  ·  │
└───────────┴───────────┘
             │
        Curve (1, 5)
        Nodes here belong to (1, 5)
        Shared by both surface meshes
```

This is also why `getNodes(dim, tag)` works — you're asking for nodes classified on a specific BRep entity. And why `getNodes(dim=2, tag=3)` returns only the **interior** nodes of surface 3, not the boundary nodes (those belong to the bounding curves). To include boundary nodes, pass `includeBoundary=True`.

### Dual query — nodes and elements on the same entity

The classified mesh can be queried through **both** `getNodes` and `getElements` at any `(dim, tag)`. Both paths access the same mesh, from different angles:

| Query                          | Returns                                        |
| ------------------------------ | ---------------------------------------------- |
| `getNodes(dim, tag)`           | Node tags + coordinates classified on entity    |
| `getElements(dim, tag)`        | Element types + IDs + connectivity on entity    |

This works uniformly across all dimensions:

```python
# Dim 0 — a point entity
getNodes(0, pt1)       # → the node tag + coordinates at that point
getElements(0, pt1)    # → point element (type 15) wrapping that same node

# Dim 1 — a curve entity
getNodes(1, crv5)      # → nodes along the curve
getElements(1, crv5)   # → line elements (type 1) connecting those nodes

# Dim 2 — a surface entity
getNodes(2, srf3)      # → interior nodes on the surface
getElements(2, srf3)   # → tri/quad elements with connectivity referencing those nodes

# Dim 3 — a volume entity
getNodes(3, vol1)      # → interior nodes in the volume
getElements(3, vol1)   # → tet/hex elements filling that volume
```

Nodes give you **positions**. Elements give you **connectivity**. Both are classified on the same BRep entity and are always consistent.

> [!note]
> `getNodes(dim, tag)` returns only nodes classified **directly** on that entity — not the boundary. `getElements(dim, tag)` returns elements on that entity whose connectivity **references** boundary nodes. So elements on a surface connect to nodes on the bounding curves, but those curve nodes won't appear in `getNodes(2, tag)` unless you pass `includeBoundary=True`.

```python
# Interior nodes only — classified directly on surface 3
interior_tags, interior_coords, _ = gmsh.model.mesh.getNodes(2, 3)

# All nodes — including those on bounding curves and corner points
all_tags, all_coords, _ = gmsh.model.mesh.getNodes(2, 3, includeBoundary=True)
```

---

## Meshing Per Dimension

Gmsh meshes **dimension by dimension**, bottom-up. Lower dimensions must be meshed before higher dimensions, because the higher-dimensional mesh depends on the lower-dimensional discretization as its boundary.

```python
gmsh.model.mesh.generate(3)
# Internally does: generate(1) → generate(2) → generate(3)
```

Calling `generate(3)` triggers the full chain. Calling `generate(2)` meshes only curves and surfaces. You can also mesh one dimension at a time for finer control.

### Dim 0 — Points

Points get **point elements** (Gmsh type code 15, alias `point1`). This is a single node wrapped in an element.

Why does this exist? It's not about meshing — a point has no extent to discretize. Point elements exist so that **physical groups at dim 0 can contain mesh elements**, not just bare nodes. This matters for:

- Concentrated loads applied at a point
- Point springs or dashpots
- Concentrated masses
- Pinned/fixed boundary conditions at specific nodes

Without point elements, a physical group at dim 0 would have no elements to query via `getElements()`. The point element is Gmsh's way of making the dim 0 → mesh → solver pipeline uniform across all dimensions.

```python
# A physical group at dim 0 works like any other
gmsh.model.addPhysicalGroup(0, [pt1, pt2], name="BC_Pinned")

# After meshing, you can get the point elements
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(0, pt1)
# elemTypes = [15]  ← point element
# elemNodeTags = [[node_tag]]  ← the single node at that point
```

### Dim 1 — Curves

Curves are discretized into **line elements** (type 1 `line2` for linear, type 8 `line3` for quadratic). The number of elements on a curve is controlled by the mesh size at its endpoints and the size field along its length.

This is the foundation — the 1D mesh defines the edge discretization that surfaces must conform to.

### Dim 2 — Surfaces

Surfaces are meshed with **triangles** (type 2 `tri3`) and/or **quadrilaterals** (type 3 `quad4`) for linear elements. The surface mesh is bounded by the 1D mesh on its edges.

The meshing algorithm (Delaunay, Frontal-Delaunay, etc.) generates the interior nodes and connectivity. Recombination can convert triangles to quads.

### Dim 3 — Volumes

Volumes are filled with **tetrahedra** (type 4 `tet4`), **hexahedra** (type 5 `hex8`), **prisms** (type 6 `prism6`), or **pyramids** (type 7 `pyramid5`). The volume mesh is bounded by the 2D mesh on its faces.

Tet meshing is fully automatic. Hex meshing requires structured (transfinite) setup or recombination from tets. Prisms and pyramids appear as transitions between hex and tet regions.

---

## Default Meshing Parameters

When you call `gmsh.model.mesh.generate()` with zero configuration, these are Gmsh's defaults (from the source, `DefaultOptions.h`):

### Algorithms

| Option               | Default                   | Meaning                          |
| -------------------- | ------------------------- | -------------------------------- |
| `Mesh.Algorithm`     | `6` (Frontal-Delaunay)    | 2D surface meshing algorithm     |
| `Mesh.Algorithm3D`   | `1` (Delaunay)            | 3D volume meshing algorithm      |
| `Mesh.ElementOrder`  | `1` (linear)              | Polynomial order of elements     |
| `Mesh.Smoothing`     | `1`                       | Number of smoothing passes       |
| `Mesh.RecombineAll`  | `0` (off)                 | Don't recombine tris→quads       |

### Size control

| Option                            | Default  | Meaning                                           |
| --------------------------------- | -------- | ------------------------------------------------- |
| `Mesh.MeshSizeMin`               | `0`      | No minimum (effectively unlimited)                |
| `Mesh.MeshSizeMax`               | `1e22`   | No maximum (effectively unlimited)                |
| `Mesh.MeshSizeFactor`            | `1.0`    | Global multiplier applied to all sizes            |
| `Mesh.MeshSizeFromPoints`        | `1` (on) | Use sizes prescribed at geometry points           |
| `Mesh.MeshSizeFromCurvature`     | `0` (off)| Don't auto-size from curvature                   |
| `Mesh.MeshSizeExtendFromBoundary`| `1` (on) | Propagate boundary sizes into interior            |

### What this means in practice

With zero configuration, Gmsh:

1. Uses the **mesh sizes you set at geometry points** (`meshSize` parameter in `addPoint`) as the primary size control
2. **Extends** those sizes from boundaries into the interior of surfaces and volumes
3. Does **not** adapt to curvature (you get uniform-ish elements even around tight curves)
4. Applies **no global min/max** — sizes are purely driven by geometry point prescriptions
5. If no sizes are prescribed anywhere, falls back to a size derived from the **model bounding box**

For structural analysis, you almost always need to override at least the mesh size. Relying on point-prescribed sizes alone gives poor control over interior refinement.

---

## Mesh Size Control Hierarchy

This is the core of "why is my mesh this size?" Gmsh resolves the final element size at any point by combining multiple sources. The logic is in `BGM_MeshSize()` (from `BackgroundMeshTools.cpp`):

### Step 1 — Collect candidate sizes

Five sources contribute a candidate size, and Gmsh takes the **minimum** of all active sources:

```
l1 ← sizes from geometry points       (if MeshSizeFromPoints = 1)
l2 ← sizes from curvature             (if MeshSizeFromCurvature > 0)
l3 ← sizes from background field      (if a field is set)
l4 ← per-entity mesh size             (if set via API)
l5 ← prescribed size at parametric point (for curves)

lc = min(l1, l2, l3, l4, l5)
```

If none of these yield a value, the fallback is the **model characteristic length** — roughly the diagonal of the bounding box.

### Step 2 — Clamp to global bounds

```
lc = max(lc, MeshSizeMin)
lc = min(lc, MeshSizeMax)
```

### Step 3 — Apply scale factors

```
lc = lc × entity_size_factor    (per-entity, if set)
lc = lc × MeshSizeFactor        (global)
```

### The full pipeline as a diagram

```
    ┌──────────────────┐
    │ Point sizes (l1) │──┐
    └──────────────────┘  │
    ┌──────────────────┐  │
    │ Curvature  (l2)  │──┤
    └──────────────────┘  │     ┌─────────┐     ┌───────────┐     ┌──────────────┐
    ┌──────────────────┐  ├──►  │ min(all) │──►  │ clamp to  │──►  │ × SizeFactor │──► final lc
    │ Fields     (l3)  │──┤     └─────────┘     │ [Min,Max] │     │ (entity+global)│
    └──────────────────┘  │                      └───────────┘     └──────────────┘
    ┌──────────────────┐  │
    │ Entity size (l4) │──┤
    └──────────────────┘  │
    ┌──────────────────┐  │
    │ Param. pts  (l5) │──┘
    └──────────────────┘
```

> [!important]
> The combination rule is **minimum, then clamp, then scale**. This means a background field can only make the mesh **finer** than what geometry points prescribe — it cannot coarsen it. To override point sizes, use `MeshSizeFromPoints = 0` or set `MeshSizeMax`.

### Boundary size extension

`MeshSizeExtendFromBoundary` controls whether sizes from lower-dimensional entities propagate inward:

- `1` (default) — extend from curves into surfaces, and from surfaces into volumes
- `0` — don't extend; interior sizing relies entirely on fields or the model characteristic length
- `2` — like 1, but uses smallest (not longest) surface edge length for 3D Delaunay
- `-2` — extend only into surfaces (not volumes)
- `-3` — extend only into volumes (not surfaces)

This is why changing the size at a geometry point affects the mesh deep inside a volume — the size propagates inward from the boundary.

---

## Meshing Workflow

The intended workflow in Gmsh, from geometry to final mesh:

### 1. Set global defaults

```python
gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
gmsh.option.setNumber("Mesh.Algorithm", 6)        # 2D algorithm
gmsh.option.setNumber("Mesh.Algorithm3D", 1)       # 3D algorithm
```

### 2. Set per-entity sizes (optional)

```python
gmsh.model.mesh.setSize([(0, pt1), (0, pt2)], target_size)
# Sets mesh size at geometry points — propagates along adjacent curves/surfaces
```

### 3. Define size fields for local refinement (optional)

```python
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [edge_tag])

f_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", fine)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", coarse)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", r_fine)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", r_coarse)
gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
```

### 4. Generate

```python
gmsh.model.mesh.generate(3)   # generates dim 1 → 2 → 3
```

### 5. Optimize (optional but recommended)

```python
gmsh.model.mesh.optimize()
# or specific methods:
gmsh.model.mesh.optimize("Netgen")          # Netgen optimizer for tets
gmsh.model.mesh.optimize("HighOrder")       # for quadratic+ elements
gmsh.model.mesh.optimize("Relocate3D")      # node relocation
```

### 6. Set element order (if not linear)

```python
gmsh.model.mesh.setOrder(2)   # convert to quadratic elements
```

> [!note]
> Element order can be set before or after generation. If set before, Gmsh generates high-order elements directly. If set after, it converts the linear mesh to high-order by adding mid-edge (and mid-face, mid-volume) nodes.

---

## Structured vs Unstructured Meshing

These are two fundamentally different approaches to filling a domain with elements.

### Unstructured

The default. Gmsh's algorithms (Delaunay, Frontal-Delaunay, etc.) fill the domain automatically. You control element size, but not element placement or topology.

- Produces **triangles** (2D) and **tetrahedra** (3D) by default
- Works on **any** geometry — no topological constraints
- Element count and arrangement are non-deterministic
- Recombination can convert tris→quads, tets→hexes (with mixed quality)

### Structured (Transfinite)

You explicitly prescribe the mesh topology. The surface or volume is mapped to a regular grid.

```python
# 1. Prescribe node count on each curve
gmsh.model.mesh.setTransfiniteCurve(tag, numNodes, meshType="Progression", coef=1.0)

# 2. Declare surface as transfinite (must have 3 or 4 bounding curves)
gmsh.model.mesh.setTransfiniteSurface(tag, arrangement="Left", cornerTags=[])

# 3. Declare volume as transfinite (must have 6 or 8 bounding surfaces)
gmsh.model.mesh.setTransfiniteVolume(tag, cornerTags=[])

# 4. Recombine to get quads/hexes
gmsh.model.mesh.setRecombine(2, surfTag)
gmsh.model.mesh.setRecombine(3, volTag)
```

Transfinite requirements:

- Surfaces must have **3 or 4 corner points** with matching node counts on opposite edges
- Volumes must have topologically mappable shapes (e.g., a box with six faces)
- All bounding curves must have `setTransfiniteCurve` applied

Transfinite gives:
- **Pure quad/hex** meshes (with recombine)
- Deterministic, reproducible meshes
- Control over element grading (via `Progression` or `Bump`)
- But only works on simple topologies — no arbitrary shapes

### When to use which

| Approach     | Geometry         | Element quality | Control      |
| ------------ | ---------------- | --------------- | ------------ |
| Unstructured | Anything         | Variable        | Size only    |
| Structured   | Simple topology  | High            | Full         |
| Hybrid       | Mixed complexity | Mixed           | Per-entity   |

For structural analysis, the hybrid approach is common: structured meshing on regular regions (beam flanges, plate sections) and unstructured on complex regions (connections, fillets), with transition elements at the interface.

---

## Mesh Output

After generation, the mesh is accessible through the API as two parallel data structures: **nodes** and **elements**.

### Nodes — position data

```python
nodeTags, nodeCoords, parametricCoord = gmsh.model.mesh.getNodes(
    dim=-1, tag=-1, includeBoundary=False, returnParametricCoord=True
)
```

| Return            | Type              | Shape / Layout                       |
| ----------------- | ----------------- | ------------------------------------ |
| `nodeTags`        | `ndarray[int64]`  | `(N,)` — globally unique node IDs   |
| `nodeCoords`      | `ndarray[float64]`| `(3N,)` — flat: $[x_1, y_1, z_1, x_2, y_2, z_2, \dots]$ |
| `parametricCoord` | `ndarray[float64]`| parametric position on owning entity |

To get a usable coordinate array:

```python
coords = nodeCoords.reshape(-1, 3)   # shape (N, 3)
# coords[i] = [x_i, y_i, z_i]
```

> [!warning]
> `nodeTags` are **not necessarily contiguous** or starting from 1. You might get `[1, 2, 5, 8, 12, ...]`. When building solver models, create a tag→index mapping:
> ```python
> tag_to_idx = {tag: i for i, tag in enumerate(nodeTags)}
> ```

### Elements — connectivity data

```python
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=-1, tag=-1)
```

The return is **grouped by element type**:

| Return          | Type                         | Structure                          |
| --------------- | ---------------------------- | ---------------------------------- |
| `elemTypes`     | `list[int]`                  | Gmsh type codes, one per type      |
| `elemTags`      | `list[ndarray[int64]]`       | Element IDs, one array per type    |
| `elemNodeTags`  | `list[ndarray[int64]]`       | Flat connectivity, one array per type |

To parse the connectivity:

```python
for etype, etags, enodes in zip(elemTypes, elemTags, elemNodeTags):
    name, dim, order, npe, _, _ = gmsh.model.mesh.getElementProperties(etype)
    conn = enodes.reshape(-1, npe)    # shape (n_elements, nodes_per_element)
    
    # conn[i] = [node_tag_1, node_tag_2, ..., node_tag_npe]
    # These are node TAGS, not indices — use tag_to_idx to convert
```

### Putting it together — the full extraction

```python
# 1. Get all nodes
nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
coords = nodeCoords.reshape(-1, 3)
tag_to_idx = {int(t): i for i, t in enumerate(nodeTags)}

# 2. Get all elements
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()

# 3. Parse each element type
for etype, etags, enodes in zip(elemTypes, elemTags, elemNodeTags):
    name, dim, order, npe, _, _ = gmsh.model.mesh.getElementProperties(etype)
    conn = enodes.reshape(-1, npe)
    
    print(f"Type: {name} (code {etype}), dim={dim}, order={order}, "
          f"npe={npe}, count={len(etags)}")
    
    # For each element: ID + connectivity as node tags
    for i, eid in enumerate(etags):
        node_tags = conn[i]          # node tags for this element
        node_indices = [tag_to_idx[int(n)] for n in node_tags]  # convert to indices
        node_xyz = coords[node_indices]  # (npe, 3) coordinate block
```

### Filtering by entity or physical group

You don't have to get everything at once. The `dim` and `tag` parameters filter by BRep entity:

```python
# Elements on surface 5 only
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2, tag=5)

# Nodes in physical group "BC_Fixed" (dim 2, pg tag 1)
nodeTags, nodeCoords = gmsh.model.mesh.getNodesForPhysicalGroup(dim=2, tag=1)
```

### Summary — data flow

```
BRep entity (dim, tag)
    │
    │  generate()
    ▼
Nodes:    tag → (x, y, z)              ← getNodes()
Elements: tag → (type, [node_tags])     ← getElements()
    │
    │  getElementProperties(type)
    ▼
Type metadata: code → (name, dim, order, npe)
    │
    │  reshape + tag mapping
    ▼
Solver-ready:  element_id, [node_indices], coordinates
```

The mesh output is always the same three things: **where are the nodes** (coordinates), **what are the elements** (type + connectivity as node tags), and **what does each element type mean** (dimension, order, nodes per element). Everything else — physical group labels, entity classification, parametric coordinates — is metadata layered on top.
