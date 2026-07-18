# Gmsh Meshing Advanced

Advanced meshing topics: algorithms, quad generation, size fields, optimization, high-order elements, embedded/compound entities, node coherence, partitioning, and quality metrics.

---

## Meshing Algorithms

Gmsh selects algorithms independently for 1D, 2D, and 3D. The 1D algorithm is fixed (equidistant subdivision of curves with size control). The 2D and 3D algorithms are configurable.

### 2D Algorithms

Set via `Mesh.Algorithm` (default: `6`, Frontal-Delaunay).

| Code | Name                       | Output      | Notes                                           |
| ---- | -------------------------- | ----------- | ----------------------------------------------- |
| 1    | MeshAdapt                  | Triangles   | Adaptive, good for curved surfaces              |
| 2    | Automatic                  | Triangles   | Selects between Delaunay and MeshAdapt           |
| 5    | Delaunay                   | Triangles   | Fast, robust for flat/simple surfaces            |
| 6    | Frontal-Delaunay           | Triangles   | Best overall quality for triangles (default)     |
| 7    | BAMG                       | Triangles   | Anisotropic adaptation                           |
| 8    | Frontal-Delaunay for Quads | Quads+Tris  | Direct quad-dominant generation                  |
| 9    | Packing of Parallelograms  | Quads       | Experimental, pure quad                          |
| 11   | Quasi-Structured Quad      | Quads       | Experimental, quasi-structured quad              |

For structural analysis, algorithm `6` (Frontal-Delaunay) is the safe default for triangular meshes. For quad-dominant meshes, algorithm `8` generates quads directly rather than recombining triangles.

```python
gmsh.option.setNumber("Mesh.Algorithm", 6)   # global
gmsh.model.mesh.setAlgorithm(2, surfTag, 8)  # per-entity override
```

### 3D Algorithms

Set via `Mesh.Algorithm3D` (default: `1`, Delaunay).

| Code | Name         | Output      | Notes                                          |
| ---- | ------------ | ----------- | ---------------------------------------------- |
| 1    | Delaunay     | Tetrahedra  | Default, robust                                |
| 4    | Frontal      | Tetrahedra  | Better quality but slower                      |
| 7    | MMG3D        | Tetrahedra  | Remeshing/adaptation (needs MMG library)       |
| 10   | HXT          | Tetrahedra  | Parallel, fast for large models                |

All 3D algorithms produce tetrahedra. Hex generation requires either structured meshing (transfinite) or recombination (experimental).

```python
gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT for large models
```

### Algorithm selection for structural analysis

| Scenario                        | 2D       | 3D       |
| ------------------------------- | -------- | -------- |
| General purpose                 | 6        | 1        |
| Large models (speed)            | 5        | 10 (HXT) |
| Quad-dominant surfaces          | 8        | —        |
| Curved geometry (shells)        | 1        | —        |
| Anisotropic refinement          | 7 (BAMG) | —        |

---

## Quad Meshing from Unstructured

Gmsh's unstructured algorithms produce triangles (2D) and tetrahedra (3D) by default. There are three paths to getting quadrilateral elements:

### Path 1 — Recombination (post-process)

Generate triangles first, then combine pairs of triangles into quads:

```python
gmsh.model.mesh.setRecombine(2, surfTag)   # per surface
# or globally:
gmsh.option.setNumber("Mesh.RecombineAll", 1)
```

This uses the **Blossom** algorithm (default) to find optimal triangle pairings. Results are quad-dominant, not pure quad — some triangles may remain where pairing isn't possible.

Recombination settings:

```python
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
# 0 = simple, 1 = blossom (default), 2 = simple full-quad, 3 = blossom full-quad

gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)   # topology cleanup passes
gmsh.option.setNumber("Mesh.RecombineNodeRepositioning", 1)  # allow node movement
gmsh.option.setNumber("Mesh.RecombineMinimumQuality", 0.01)  # min quad quality
```

### Path 2 — Direct quad algorithm

Use a 2D algorithm that generates quads natively:

```python
gmsh.option.setNumber("Mesh.Algorithm", 8)   # Frontal-Delaunay for Quads
```

This produces quads during generation rather than converting after. Often gives better quality than recombination, but is less robust on complex geometry.

### Path 3 — Subdivision

Generate a triangular mesh, then subdivide each triangle into 3 quads (or each quad into 4 quads):

```python
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # all-quads
# 0 = none, 1 = all-quads, 2 = all-hexas (3D)
```

This guarantees pure quads but approximately triples the element count and may produce poor quality elements at boundaries.

### 3D recombination

Recombining tets into hexahedra is experimental:

```python
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)
# 0 = hex only, 1 = hex+prisms, 2 = hex+prisms+pyramids
```

> [!warning]
> 3D recombination is unreliable for production use. For hex-dominant 3D meshes, use structured (transfinite) meshing on suitable geometry.

---

## Fields

Fields are spatial functions that control mesh element size. They are the primary mechanism for **local refinement** — making the mesh finer near features of interest and coarser away from them.

### How fields work

A field is a function $f(x, y, z) \to \text{size}$. When set as the background mesh, it participates in the size resolution pipeline (see `gmsh_meshing_basics.md` — Step 1, source `l3`). The minimum of the field value and all other size sources determines the final element size.

### The Distance + Threshold pattern

This is the most common pattern for structural refinement. Two fields work together:

```python
# 1. Distance field — measures distance to target entities
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [edge1, edge2])
gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

# 2. Threshold field — maps distance to mesh size
f_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", 0.5)     # fine size (near)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", 5.0)     # coarse size (far)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 1.0)     # start transition
gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 10.0)    # end transition

gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
```

```
Size
 ▲
 │  SizeMax ─────────────────────────────
 │                              ╱
 │                            ╱
 │                          ╱   transition
 │                        ╱
 │  SizeMin ─────────────
 │          │            │              │
 └──────────┼────────────┼──────────────┼──► Distance
            0        DistMin        DistMax
```

### Combining multiple fields

When you need refinement around multiple features, create separate Distance+Threshold pairs and combine with `Min`:

```python
f_min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thresh1, f_thresh2, f_thresh3])
gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
```

`Min` takes the smallest size from all input fields at each point — so each refinement zone is respected.

### Available field types

| Type          | Description                                              |
| ------------- | -------------------------------------------------------- |
| `Distance`    | Distance to points, curves, or surfaces                  |
| `Threshold`   | Maps distance (from another field) to size via ramp      |
| `Min` / `Max` | Takes min/max of multiple fields                         |
| `MathEval`    | Arbitrary expression $f(x,y,z)$ as a string              |
| `Box`         | Uniform size inside a box, different outside              |
| `Cylinder`    | Uniform size inside a cylinder                           |
| `Constant`    | Uniform size everywhere                                  |
| `BoundaryLayer` | Structured boundary layer (prisms/hexes along surfaces)|
| `Restrict`    | Restrict a field to specific entities                    |

### MathEval — arbitrary expressions

```python
f = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(f, "F", "0.1 + 0.05 * sqrt(x*x + y*y)")
```

Useful for radially graded meshes, linear gradients, or any analytically defined size distribution.

> [!note]
> Only **one** field can be the background mesh at a time. To use multiple fields, combine them with `Min` (or `Max`) and set the combined field as background.

---

## Mesh Optimization

After generation, mesh quality can be improved through optimization passes. The `optimize()` function supports several methods:

```python
gmsh.model.mesh.optimize(method="", force=False, niter=1, dimTags=[])
```

### Available methods

| Method                   | Applies to | What it does                                         |
| ------------------------ | ---------- | ---------------------------------------------------- |
| `""` (empty string)      | 3D tets    | Default tet optimizer (edge/face swaps + smoothing)   |
| `"Netgen"`               | 3D tets    | Netgen optimizer (often better than default)          |
| `"Laplace2D"`            | 2D         | Laplacian smoothing (moves nodes to avg of neighbors) |
| `"Relocate2D"`           | 2D         | Node relocation for quality improvement              |
| `"Relocate3D"`           | 3D         | Node relocation for quality improvement              |
| `"HighOrder"`            | Any        | Optimize high-order element validity                  |
| `"HighOrderElastic"`     | Any        | Elastic smoothing for high-order elements             |
| `"HighOrderFastCurving"` | Any        | Fast curving for high-order near boundaries           |
| `"QuadQuasiStructured"`  | 2D quads   | Improve quad mesh topology                            |
| `"UntangleMeshGeometry"` | Any        | Fix inverted elements                                 |

### Typical optimization sequence

```python
gmsh.model.mesh.generate(3)

# Optimize tets
gmsh.model.mesh.optimize("")           # default optimizer
gmsh.model.mesh.optimize("Netgen")     # additional Netgen pass

# If using high-order elements
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("HighOrder")  # fix invalid high-order elements
```

Multiple passes can be applied (`niter` parameter or repeated calls). The `dimTags` parameter restricts optimization to specific entities.

> [!note]
> Optimization moves nodes and may swap element faces/edges, but never changes the element count or topology fundamentally. It works within the existing mesh structure.

---

## High-Order Elements

By default, Gmsh generates linear (order 1) elements. Higher-order elements add mid-edge, mid-face, and mid-volume nodes for better geometric and solution accuracy.

```python
gmsh.model.mesh.setOrder(2)    # quadratic
gmsh.model.mesh.setOrder(3)    # cubic
```

### Straight vs curved high-order

By default, high-order nodes are placed on the actual geometry (curved elements). This is controlled by:

```python
gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # 0 = curved (default), 1 = straight
```

**Curved** elements follow the CAD surface exactly — important for shells and surfaces with curvature. **Straight** elements place mid-nodes at the midpoint of the straight edge — simpler but loses geometric accuracy.

### Serendipity vs Lagrangian

For quadratic elements:

```python
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
# 0 = complete (Lagrangian) — e.g., quad9 with center node
# 1 = incomplete (serendipity) — e.g., quad8 without center node
```

Serendipity elements (incomplete) are more common in structural analysis — they have fewer nodes and no center node on faces, which avoids some numerical issues.

### High-order validity

Curved high-order elements can become invalid (negative Jacobian) if the curvature is too high relative to the element size. The optimization methods `"HighOrder"`, `"HighOrderElastic"`, and `"HighOrderFastCurving"` fix these:

```python
gmsh.model.mesh.setOrder(2)
gmsh.model.mesh.optimize("HighOrder")

# Or via option for automatic optimization:
gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
# 0 = none, 1 = optimization, 2 = elastic+optimization, 3 = elastic, 4 = fast curving
```

### Relevance for structural analysis

- **Linear elements (order 1):** standard for OpenSees. Simple, robust, but may suffer from locking (shear locking in bending, volumetric locking in near-incompressible materials).
- **Quadratic elements (order 2):** better accuracy per DOF, reduced locking. Require compatible solver element formulations.
- **Cubic+ (order 3+):** rarely used in structural practice. High cost, diminishing returns.

---

## Embedded Entities

Embedding forces the mesh to conform to a lower-dimensional entity that lies **inside** (not on the boundary of) a higher-dimensional entity.

```python
gmsh.model.mesh.embed(dim, tags, inDim, inTag)
# dim    — dimension of entities to embed (0, 1, or 2)
# tags   — tags of entities to embed
# inDim  — dimension of the host entity (2 or 3), must be > dim
# inTag  — tag of the host entity
```

Examples:

```python
# Embed a point in a surface — forces a mesh node at that point
gmsh.model.mesh.embed(0, [pt1], 2, surfTag)

# Embed a curve in a surface — forces element edges along the curve
gmsh.model.mesh.embed(1, [crv1], 2, surfTag)

# Embed a curve in a volume — forces element edges along the curve
gmsh.model.mesh.embed(1, [crv1], 3, volTag)

# Embed a surface in a volume — forces element faces on the surface
gmsh.model.mesh.embed(2, [srf1], 3, volTag)
```

### Use cases in structural analysis

- **Rebar lines** embedded in a concrete volume — mesh conforms to the rebar path
- **Crack paths** embedded in a surface — element edges follow the crack
- **Load application lines** embedded in a surface — nodes guaranteed along the load path
- **Sensor points** embedded in a volume — nodes at exact measurement locations

### Embed vs Fragment

Both achieve mesh conformity, but at different levels:

- **`embed`** — mesh-level constraint. The host entity's mesh conforms to the embedded entity's mesh. The BRep topology is unchanged.
- **`fragment`** — geometry-level operation. Splits the BRep at intersections, creating new entities. The resulting mesh is naturally conformal because the entities share boundaries.

With the OCC kernel, `fragment` applied to entities of different dimensions **automatically embeds** the lower-dimensional entities in the higher-dimensional ones if they're not already on the boundary. So in OCC, fragment is often the preferred approach.

```python
# OCC: fragment automatically embeds the curve in the volume
out, out_map = gmsh.model.occ.fragment([(3, vol)], [(1, crv)])
# The curve is now embedded in the volume — mesh will conform
```

---

## Compound Entities

Compound meshing treats multiple entities of the same dimension as a **single entity** for meshing purposes. The internal boundaries between them are ignored — the mesher sees one continuous surface or volume.

```python
gmsh.model.mesh.setCompound(dim, tags)
# dim  — dimension of entities to compound (1 or 2)
# tags — list of entity tags to merge for meshing
```

### Why it matters

CAD imports often produce surfaces that are fragmented into many small patches (e.g., a cylinder split into four quarter-surfaces). Each patch is meshed independently, and the mesh must conform at the patch boundaries — this can force poor element quality at the seams.

Compound meshing removes these artificial constraints:

```python
# CAD imported a cylinder as 4 surface patches: [1, 2, 3, 4]
# Mesh them as a single surface:
gmsh.model.mesh.setCompound(2, [1, 2, 3, 4])
```

The mesher reparametrizes the compound as a single discrete entity, producing a smoother mesh across the original patch boundaries.

> [!note]
> Compound meshing changes how the entity is discretized but doesn't modify the BRep topology. The original entities still exist for physical group assignment and other queries.

---

## Node Duplication and Mesh Coherence

### The problem

When two entities are meshed independently and share a geometric boundary, the question is: do they share nodes at that boundary, or does each get its own copy?

**Conformal mesh** — shared nodes at the interface. Structural continuity is automatic. This is the normal case when entities share BRep boundaries (because nodes are classified on the shared boundary entity).

**Non-conformal mesh** — duplicate nodes at the interface. Each entity has its own nodes. No structural continuity unless explicitly tied.

### When nodes are shared (conformal)

Nodes are shared automatically when entities share BRep boundaries. This is the default behavior:

```
Volume A and Volume B share Surface S
    ↓
Surface S is meshed once
    ↓
Nodes on Surface S are shared by both volumes
    ↓
Conformal interface — no duplicate nodes
```

This is why `fragment` is critical for multi-material models: it makes two overlapping volumes share a common interface surface in the BRep, which guarantees shared nodes.

### When nodes are duplicated (non-conformal)

Duplicate nodes appear when:

1. **No shared BRep boundary** — two volumes that touch geometrically but don't share a surface in the BRep. This happens when you create two boxes that share a face but don't `fragment` them.
2. **Mesh import** — importing a mesh where nodes were independently numbered per region.
3. **Intentional duplication** — for cohesive zone elements, contact interfaces, or discontinuities where you *want* separate nodes at the same location.

### Fragment as the geometry-level solution

`fragment` is the primary tool for ensuring mesh conformity:

```python
# Without fragment: two boxes touching but not sharing a BRep face
box1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
box2 = gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
# → duplicate nodes at x=1 plane — no structural continuity

# With fragment: BRep is split, interface surface is shared
out, out_map = gmsh.model.occ.fragment([(3, box1)], [(3, box2)])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
# → shared nodes at x=1 plane — conformal interface
```

### `removeDuplicateNodes` as the mesh-level fix

If you already have duplicate nodes (from import or non-fragmented geometry), you can merge them:

```python
gmsh.model.mesh.removeDuplicateNodes()
# Merges nodes within the default geometric tolerance
```

This finds nodes at the same coordinates (within tolerance) and merges them, updating element connectivity to reference the surviving node.

> [!warning]
> `removeDuplicateNodes` is a post-hoc fix. It works on coordinates, not topology — so it can accidentally merge nodes that happen to be at the same location but shouldn't be connected (e.g., at a hinge or contact interface). Prefer `fragment` at the geometry level when possible.

### `removeDuplicateElements`

Similarly, duplicate elements (same connectivity) can be removed:

```python
gmsh.model.mesh.removeDuplicateElements()
```

### Intentional duplication — cohesive zones and contact

Sometimes you **want** duplicate nodes: at a cohesive zone interface where zero-thickness elements connect two surfaces, or at a contact interface where surfaces can separate. In these cases:

- Do **not** fragment the geometry — keep separate BRep surfaces
- Do **not** call `removeDuplicateNodes`
- The duplicate nodes are intentional and will be connected by interface elements in the solver

---

## Orphan Nodes

Orphan nodes are nodes that exist in the mesh but are **not referenced by any element**. They have coordinates and a tag, but no element's connectivity includes them.

### How they arise

- **Mesh editing** — deleting elements without deleting their nodes
- **Physical group filtering** — when `SaveAll = 0`, elements outside physical groups are not exported, but their nodes may persist
- **Boolean operations** — geometry changes that remove elements but leave behind classified nodes
- **Manual node addition** — calling `addNodes()` without creating elements that reference them

### Why they matter

For solver pipelines, orphan nodes create problems:

- Extra DOFs in the system matrix for nodes not connected to any element
- Singular stiffness matrix — unconstrained DOFs with no element stiffness contribution
- Incorrect node counts when sizing arrays

### Detection

Gmsh doesn't provide a direct "get orphan nodes" query. Detection is done by comparing the set of all node tags against the set of node tags referenced in element connectivity:

```python
# All nodes in the mesh
allNodeTags, _, _ = gmsh.model.mesh.getNodes()
allNodes = set(int(t) for t in allNodeTags)

# All nodes referenced by elements
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
referencedNodes = set()
for enodes in elemNodeTags:
    referencedNodes.update(int(n) for n in enodes)

# Orphan nodes
orphans = allNodes - referencedNodes
```

### Removal

There's no built-in `removeOrphanNodes()` in the Gmsh API. Options:

1. **Re-extract carefully** — when building solver models, only use nodes that appear in element connectivity. Ignore the rest.
2. **Rebuild node list** — extract only referenced nodes and create a clean tag mapping.

```python
# Build solver model using only referenced nodes
tag_to_idx = {}
idx = 1
for etype, etags, enodes in zip(elemTypes, elemTags, elemNodeTags):
    _, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(etype)
    conn = enodes.reshape(-1, npe)
    for node_tag in conn.flat:
        nt = int(node_tag)
        if nt not in tag_to_idx:
            tag_to_idx[nt] = idx
            idx += 1
# tag_to_idx now contains only nodes that are part of at least one element
```

> [!note]
> apeGmsh's mesh extraction pipeline should handle orphan nodes by construction — only emit nodes that participate in element connectivity.

---

## Mesh Partitioning

See [[gmsh_partitioning]] for a deep dive on mesh partitioning, METIS algorithms, ghost elements, and the OpenSeesSP/MP pipeline.

---

## Element Quality Metrics

Gmsh can evaluate element quality using several metrics:

```python
qualities = gmsh.model.mesh.getElementQualities(elementTags, qualityName="minSICN")
```

### Available metrics

| Metric         | Description                                            | Range     | Good value |
| -------------- | ------------------------------------------------------ | --------- | ---------- |
| `minSICN`      | Minimum signed inverted condition number (default)     | [-1, 1]   | > 0.5      |
| `minSIGE`      | Minimum signed inverted gradient error                 | [-1, 1]   | > 0.5      |
| `gamma`        | Inscribed / circumscribed sphere radius ratio          | [0, 1]    | > 0.3      |
| `minDetJac`    | Minimum Jacobian determinant (adaptive)                | varies    | > 0        |
| `maxDetJac`    | Maximum Jacobian determinant (adaptive)                | varies    | > 0        |
| `minSJ`        | Minimum scaled Jacobian                                | [-1, 1]   | > 0.2      |
| `innerRadius`  | Inner radius of element                                | > 0       | —          |
| `outerRadius`  | Outer radius of element                                | > 0       | —          |
| `minIsotropy`  | Minimum isotropy measure                               | [0, 1]    | > 0.5      |
| `angleShape`   | Angle shape measure                                    | [0, 1]    | > 0.3      |
| `minEdge`      | Minimum straight edge length                           | > 0       | —          |
| `maxEdge`      | Maximum straight edge length                           | > 0       | —          |
| `volume`       | Element volume/area                                    | > 0       | —          |

### Interpreting quality for structural analysis

**Negative values** (SICN, SIGE, SJ) indicate **inverted elements** — the Jacobian is negative somewhere. These will cause solver failures. Must be zero or above.

**Near-zero positive values** indicate severely distorted elements — high aspect ratios, near-degenerate shapes. These cause:
- Poor conditioning of the stiffness matrix
- Inaccurate stress/strain results
- Slow or failed convergence in nonlinear analysis

**Practical thresholds for structural analysis:**

| Metric    | Acceptable | Good   | Excellent |
| --------- | ---------- | ------ | --------- |
| `minSICN` | > 0.1      | > 0.5  | > 0.7     |
| `gamma`   | > 0.1      | > 0.3  | > 0.5     |
| `minSJ`   | > 0.0      | > 0.2  | > 0.5     |

```python
# Check quality of all tet elements
_, elemTags, _ = gmsh.model.mesh.getElements(dim=3)
all_tet_tags = []
for etags in elemTags:
    all_tet_tags.extend(etags)

qualities = gmsh.model.mesh.getElementQualities(all_tet_tags, "minSICN")
min_q = min(qualities)
avg_q = sum(qualities) / len(qualities)
bad_count = sum(1 for q in qualities if q < 0.1)

print(f"Min quality: {min_q:.3f}, Avg: {avg_q:.3f}, "
      f"Elements below 0.1: {bad_count}/{len(qualities)}")
```

> [!important]
> Always check mesh quality before running a structural analysis. A mesh that "looks fine" visually can have a few severely distorted elements that cause convergence failure. Quality metrics catch what visual inspection misses.
