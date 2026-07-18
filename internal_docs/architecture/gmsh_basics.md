# Gmsh Basics

## BRep — Boundary Representation

Gmsh models geometry using **Boundary Representation** (BRep). Every geometric object is defined by its boundary — a volume is bounded by surfaces, a surface by curves, a curve by points.

The BRep hierarchy has four levels, indexed by **dimension**:

| dim | Entity   | Bounded by       |
| --- | -------- | ---------------- |
| 0   | Point    | —                |
| 1   | Curve    | Points (dim 0)   |
| 2   | Surface  | Curves (dim 1)   |
| 3   | Volume   | Surfaces (dim 2) |

Every entity is uniquely identified by the pair `(dim, tag)`. Tags are unique **within** a dimension — a curve with tag 5 and a surface with tag 5 are distinct entities.

```
Volume (3, 1)
├── Surface (2, 1)
│   ├── Curve (1, 1) ─── Point (0, 1), Point (0, 2)
│   ├── Curve (1, 2) ─── Point (0, 2), Point (0, 3)
│   ├── Curve (1, 3) ─── Point (0, 3), Point (0, 4)
│   └── Curve (1, 4) ─── Point (0, 4), Point (0, 1)
├── Surface (2, 2)
│   └── ...
└── ...
```

Adjacency queries navigate this tree in both directions:

```python
upward, downward = gmsh.model.getAdjacencies(dim, tag)
# upward   → entities of dim+1 that contain this entity
# downward → entities of dim-1 that bound this entity
```

### OCC Kernel

Gmsh provides two geometry kernels. Both produce the same BRep data model — they differ only in **how** geometry is constructed.

**Built-in kernel** (`gmsh.model.geo`): bottom-up construction. You explicitly create points, wire them into curves, loop curves into surfaces, shell surfaces into volumes. Full control over tags, but limited to constructive geometry.

**OpenCASCADE kernel** (`gmsh.model.occ`): high-level CAD operations. Provides solid primitives (`addBox`, `addCylinder`, ...), boolean operations (`fuse`, `cut`, `intersect`, `fragment`), fillets, chamfers, and CAD import (STEP, IGES, BREP). OCC handles the BRep topology internally — splitting, stitching, and rebuilding faces as needed.

The tradeoff:

```
            ┌─────────────┐        ┌──────────────┐
            │  geo kernel  │        │  occ kernel   │
            ├─────────────┤        ├──────────────┤
            │ Bottom-up    │        │ High-level    │
            │ Explicit tags│        │ Tags shift    │
            │ No booleans  │        │ Full booleans │
            │ No CAD import│        │ STEP/IGES/BRep│
            │ Lightweight  │        │ Needs OCC lib │
            └──────┬──────┘        └──────┬───────┘
                   │                       │
                   └────────┬──────────────┘
                            ▼
                    Same BRep model
                     (dim, tag) tree
```

> [!important]
> After any geometry operations, you **must** call `synchronize()` before meshing or querying.
> This rebuilds the internal BRep topology.
> ```python
> gmsh.model.geo.synchronize()  # or gmsh.model.occ.synchronize()
> ```

> [!warning]
> Never mix entities from different kernels in the same model. Pick one.

---

## Geometry → Mesh Paradigm

The BRep defines the **geometry**. The mesh is a discrete approximation that lives **on top of** that geometry. The relationship is:

```
    GEOMETRY (continuous)              MESH (discrete)
    ─────────────────────              ────────────────
    BRep entities                      Nodes + Elements
    (dim, tag)                         classified onto entities

    Point   (0, tag)  ─────────────►  Node(s) at that point
    Curve   (1, tag)  ─────────────►  Line elements along that curve
    Surface (2, tag)  ─────────────►  Tri/Quad elements on that face
    Volume  (3, tag)  ─────────────►  Tet/Hex elements filling that volume
```

The mesh generation pipeline:

1. **Size control** — set mesh sizes (global, per-entity, or via fields)
2. **Generate dim 1** — discretize curves into line elements
3. **Generate dim 2** — mesh surfaces with triangles/quads
4. **Generate dim 3** — fill volumes with tetrahedra/hexahedra
5. **Optimize** — improve element quality (smoothing, swapping)

### Nodes

A mesh node is identified by a single globally unique **tag** (integer). Nodes are classified onto the BRep entity they belong to, but that classification is not part of the node's identity.

```python
nodeTags, nodeCoords, parametricCoord = gmsh.model.mesh.getNodes(
    dim=-1, tag=-1, returnParametricCoord=True
)
```

| Return             | Shape                  | Description                                           |
| ------------------ | ---------------------- | ----------------------------------------------------- |
| `nodeTags`         | `(N,)`                 | Globally unique integer IDs (may not be contiguous)   |
| `nodeCoords`       | `(3N,)` → reshape `(N,3)` | Physical coordinates $[x_1, y_1, z_1, x_2, \dots]$ |
| `parametricCoord`  | varies                 | Position in entity's parameter space ($u$ or $u,v$)   |

The parametric coordinates map each node back to the BRep entity's own coordinate system:

- On a curve (dim 1): one parameter $u$
- On a surface (dim 2): two parameters $(u, v)$
- On a volume (dim 3) or point (dim 0): not meaningful

> [!note]
> Parametric coords let you evaluate geometric properties (normals, curvature, tangent vectors) at a node's location via the BRep. Rarely needed for structural analysis, but essential for the geometry-mesh link.

### Elements

An element is identified by a single globally unique **tag**. Each element has an associated **element type** (an integer code) that defines its shape, dimension, polynomial order, and number of nodes.

```python
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=-1, tag=-1)
```

The return is **grouped by element type**:

| Return          | Structure                 | Description                               |
| --------------- | ------------------------- | ----------------------------------------- |
| `elemTypes`     | `[type_a, type_b, ...]`  | List of Gmsh type codes present           |
| `elemTags`      | `[array_a, array_b, ...]` | Element IDs, one array per type          |
| `elemNodeTags`  | `[array_a, array_b, ...]` | Flat connectivity, one array per type    |

The connectivity for each type is a flat array of node tags. To get the per-element rows, you need the number of nodes per element:

```python
for etype, etags, enodes in zip(elemTypes, elemTags, elemNodeTags):
    name, dim, order, npe, _, _ = gmsh.model.mesh.getElementProperties(etype)
    conn = enodes.reshape(-1, npe)  # shape (n_elements, nodes_per_element)
```

> [!important]
> Node tags in `elemNodeTags` reference the globally unique node tags from `getNodes()`, **not** sequential indices. When building a solver model (e.g. OpenSees), create a mapping from Gmsh tags to contiguous IDs.

---

## Element Types

Every Gmsh element type is defined by a unique integer **code**. From that code you can query all metadata:

```python
name, dim, order, numNodes, localCoords, numPrimaryNodes = \
    gmsh.model.mesh.getElementProperties(elementType)
```

An element type encodes four things:

- **Shape** — the topological family (line, triangle, quad, tet, hex, prism, pyramid)
- **Dimension** — inherited from the shape (1D, 2D, 3D)
- **Order** — polynomial degree of the interpolation (1 = linear, 2 = quadratic, ...)
- **Nodes per element** — determined by shape + order

### Classification by Shape and Order

#### 1D Elements — Lines

$$n_{\text{nodes}} = \text{order} + 1$$

| Code | Alias   | Order | Nodes | Shape              |
| ---- | ------- | ----- | ----- | ------------------ |
| 1    | `line2` | 1     | 2     | `o─────────o`      |
| 8    | `line3` | 2     | 3     | `o────o────o`      |
| 26   | `line4` | 3     | 4     | `o──o──o──o`       |

Used for: trusses, beam-column elements, 1D boundary conditions.

#### 2D Elements — Triangles

| Code | Alias   | Order | Nodes | Description            |
| ---- | ------- | ----- | ----- | ---------------------- |
| 2    | `tri3`  | 1     | 3     | Linear triangle        |
| 9    | `tri6`  | 2     | 6     | Quadratic triangle     |
| 20   | `tri9`  | 3     | 9     | Cubic (incomplete)     |
| 21   | `tri10` | 3     | 10    | Cubic (complete)       |

```
tri3 (linear)          tri6 (quadratic)
    2                      2
   / \                    / \
  /   \                  5   4
 /     \                / \  / \
0───────1              0───3───1
```

#### 2D Elements — Quadrilaterals

| Code | Alias    | Order | Nodes | Description        |
| ---- | -------- | ----- | ----- | ------------------ |
| 3    | `quad4`  | 1     | 4     | Bilinear quad      |
| 16   | `quad8`  | 2     | 8     | Serendipity quad   |
| 10   | `quad9`  | 2     | 9     | Biquadratic quad   |
| 36   | `quad16` | 3     | 16    | Bicubic quad       |

```
quad4 (bilinear)       quad9 (biquadratic)
3───────2              3───6───2
|       |              |       |
|       |              7   8   5
|       |              |       |
0───────1              0───4───1
```

> [!note]
> `quad8` (serendipity) has nodes only at corners and mid-edges — no center node. `quad9` (Lagrangian) adds the center node. Serendipity is more common in structural analysis.

#### 3D Elements — Tetrahedra

| Code | Alias    | Order | Nodes | Description        |
| ---- | -------- | ----- | ----- | ------------------ |
| 4    | `tet4`   | 1     | 4     | Linear tet         |
| 11   | `tet10`  | 2     | 10    | Quadratic tet      |
| 29   | `tet20`  | 3     | 20    | Cubic tet          |

Tets are the default for 3D unstructured meshing. Easy to generate for complex geometry but less accurate per DOF than hexahedra.

#### 3D Elements — Hexahedra

| Code | Alias    | Order | Nodes | Description        |
| ---- | -------- | ----- | ----- | ------------------ |
| 5    | `hex8`   | 1     | 8     | Trilinear hex      |
| 17   | `hex20`  | 2     | 20    | Serendipity hex    |
| 12   | `hex27`  | 2     | 27    | Triquadratic hex   |
| 92   | `hex64`  | 3     | 64    | Tricubic hex       |

Hexahedra are preferred for structural accuracy but require structured or semi-structured meshing (transfinite, recombine).

#### 3D Elements — Prisms and Pyramids

| Code | Alias        | Order | Nodes | Description           |
| ---- | ------------ | ----- | ----- | --------------------- |
| 6    | `prism6`     | 1     | 6     | Linear prism (wedge)  |
| 18   | `prism15`    | 2     | 15    | Quadratic prism       |
| 7    | `pyramid5`   | 1     | 5     | Linear pyramid        |
| 19   | `pyramid13`  | 2     | 13    | Quadratic pyramid     |

Prisms and pyramids appear as **transition elements** between hex-meshed and tet-meshed regions, or in boundary layer meshes.

#### 0D Elements — Points

| Code | Alias    | Order | Nodes | Description    |
| ---- | -------- | ----- | ----- | -------------- |
| 15   | `point1` | 0     | 1     | Point element  |

Used for: concentrated masses, point loads, point springs.

### apeGmsh Alias System

In apeGmsh, element types are wrapped in `ElementTypeInfo` and given short aliases following the convention `{shape}{npe}`:

```python
# from _element_types.py
_KNOWN_ALIASES = {
    1:  'line2',    2:  'tri3',     3:  'quad4',
    4:  'tet4',     5:  'hex8',     6:  'prism6',    7:  'pyramid5',
    8:  'line3',    9:  'tri6',    10:  'quad9',
    11: 'tet10',   12:  'hex27',   15:  'point1',
    16: 'quad8',   17:  'hex20',   18:  'prism15',  19: 'pyramid13',
    ...
}
```

The `ElementTypeInfo` dataclass captures the full metadata:

```python
ElementTypeInfo(
    code=4,             # Gmsh integer code (primary key)
    name='tet4',        # apeGmsh alias
    gmsh_name='Tetrahedron 4',  # Gmsh's own name
    dim=3,              # topological dimension
    order=1,            # polynomial order
    npe=4,              # nodes per element
    count=12000         # how many in the mesh
)
```

Elements are stored in `ElementGroup` objects — homogeneous blocks where all elements share the same type. This gives rectangular `(N, npe)` connectivity arrays, which are efficient for solver loops:

```python
for group in mesh.elements:
    for eid, conn in group:
        ops.element(etype, eid, *conn, matTag)
```

---

## Physical Groups

### Identity and Purpose

Physical groups follow the same identification pattern as geometric entities: a `(dim, tag)` tuple. The name is just a label attached to that key — it's not part of the identity.

```python
tag = gmsh.model.addPhysicalGroup(dim, tags, tag=-1, name="")
```

A physical group is a **named grouping of geometric entities at a single dimension**. It maps one `(dim, tag)` to a set of geometric entity tags, all sharing that same dimension.

```
Physical Group (dim=2, tag=1, name="BC_Fixed")
    └── references geometric entities: Surface 3, Surface 7, Surface 12

Physical Group (dim=3, tag=1, name="Mat_Steel")
    └── references geometric entities: Volume 1, Volume 4
```

> [!important]
> Physical groups live at the geometry level. They reference BRep entities, not mesh elements. The mesh inherits these labels because elements are classified onto entities.

### Constraints

**Single dimension per group.** All entity tags in a physical group must share the same dimension. You cannot mix curves and surfaces in one group.

**Names are not unique.** Gmsh allows duplicate names — even within the same dimension. The true identity is `(dim, tag)`, not the name. Duplicate names are legal but create ambiguity in name-based lookups.

**Names can repeat across dimensions.** Two physical groups at different dimensions can share a name. They are distinct groups:

```python
gmsh.model.addPhysicalGroup(0, [pt1, pt2],     name="BC_Fixed")  # (0, tag_a)
gmsh.model.addPhysicalGroup(2, [surf1, surf2],  name="BC_Fixed")  # (2, tag_b)
```

### Relationship to Mesh

Physical groups do **not** affect mesh generation. The mesh exists independently — `getNodes()` and `getElements()` return everything regardless of physical group assignment.

Physical groups affect two things:

1. **Semantic labeling** — they give engineering meaning to geometric regions (material zones, boundary conditions, load surfaces). This is the primary purpose for solver pipelines.

2. **File export filtering** — with `Mesh.SaveAll = 0` (default), only elements on entities that belong to at least one physical group are written to `.msh`. With `SaveAll = 1`, everything is exported. This is a file format artifact, not a fundamental limitation.

When extracting mesh data through the API (which is what apeGmsh does), the export filter is irrelevant. Physical groups are purely a semantic layer:

```python
# Get nodes for a specific physical group
nodeTags, nodeCoords = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)

# Get entities in a physical group, then query their elements
entityTags = gmsh.model.getEntitiesForPhysicalGroup(dim, pgTag)
for eTag in entityTags:
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, eTag)
```

### Summary — Gmsh Data Model

The full identification scheme:

```
LAYER            IDENTITY        WHAT IT HOLDS
─────────────    ────────        ──────────────────────────
Geometric entity (dim, tag)      Shape definition (BRep)
Physical group   (dim, tag)      Set of entity tags + name
Mesh node        tag             Coordinates + parametric coords
Mesh element     tag             Type code + node connectivity
```

Physical groups are a second `(dim, tag)` namespace that references the first. Nodes and elements use flat tags. Names are metadata, not keys.

---

## Session Lifecycle

Gmsh operates as a **singleton global state machine**. There is no Gmsh object you instantiate — you initialize a global session, work with it, and finalize when done.

```python
import gmsh

gmsh.initialize()       # start the session (required before anything)
# ... all work happens here ...
gmsh.finalize()         # release resources (required at the end)
```

> [!warning]
> Calling any Gmsh function before `initialize()` or after `finalize()` will raise an error. There is no way to have two independent Gmsh sessions in the same process.

### Models

Within a session, Gmsh can hold **multiple models**. Each model has its own BRep, mesh, and physical groups. Only one model is active at a time.

```python
gmsh.model.add("beam")           # create and activate a new model
gmsh.model.add("column")         # create another — now active
gmsh.model.setCurrent("beam")    # switch back
gmsh.model.list()                # → ["beam", "column"]
```

All `gmsh.model.*` calls operate on the **current** model. There's no handle or reference — it's implicit global state. This means you can't work on two models in parallel within the same process.

```
Session (singleton)
├── Model "beam"     ← setCurrent("beam")
│   ├── BRep entities
│   ├── Physical groups
│   └── Mesh
├── Model "column"
│   ├── BRep entities
│   ├── Physical groups
│   └── Mesh
└── Global options
```

### Typical session structure

```python
import gmsh
import sys

gmsh.initialize(sys.argv)        # sys.argv passes CLI flags to Gmsh
gmsh.model.add("my_model")

# --- geometry ---
gmsh.model.occ.synchronize()

# --- physical groups ---
# --- mesh ---
# --- extract or export ---

gmsh.finalize()
```

Passing `sys.argv` allows Gmsh to pick up command-line flags (e.g., `-nopopup` to suppress the GUI).

---

## Entity Queries

Once the BRep is built and synchronized, you can navigate it programmatically. These queries work on the geometry — they don't require a mesh.

### Listing entities

```python
entities = gmsh.model.getEntities(dim=-1)
# Returns [(dim, tag), ...] for all entities
# Pass dim=2 to get only surfaces, etc.
```

### Adjacency traversal

The BRep is a graph. `getAdjacencies` walks it in both directions from a given entity:

```python
upward, downward = gmsh.model.getAdjacencies(dim, tag)
# upward   → tags of entities at dim+1 that contain this entity
# downward → tags of entities at dim-1 that bound this entity
```

Example — given a curve `(1, 5)`:

```python
upward, downward = gmsh.model.getAdjacencies(1, 5)
# upward   = [3, 7]     ← surfaces that use this curve
# downward = [1, 2]     ← points at each end of this curve
```

```
Surface (2, 3)    Surface (2, 7)       ← upward
       \             /
        Curve (1, 5)                   ← query target
       /             \
Point (0, 1)     Point (0, 2)          ← downward
```

### Boundary extraction

`getBoundary` returns the boundary entities of a set of entities — i.e., it steps one dimension down:

```python
boundary = gmsh.model.getBoundary(
    dimTags,            # [(dim, tag), ...]
    combined=True,      # merge shared boundaries
    oriented=False,     # include orientation signs
    recursive=False     # recurse to dim 0
)
# Returns [(dim-1, tag), ...]
```

With `combined=True`, shared internal boundaries cancel out — you get only the outer boundary. With `combined=False`, you get every boundary entity of every input entity, including shared ones.

With `recursive=True`, it walks all the way down to points (dim 0).

```python
# Get all edges bounding surface 3
edges = gmsh.model.getBoundary([(2, 3)])

# Get all points bounding a set of volumes (recursive)
points = gmsh.model.getBoundary([(3, 1), (3, 2)], recursive=True)
```

### Bounding box queries

```python
# Bounding box of a single entity
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

# Find all entities within a bounding box
dimTags = gmsh.model.getEntitiesInBoundingBox(
    xmin, ymin, zmin, xmax, ymax, zmax, dim=-1
)
```

These are useful for spatial queries — finding entities near a point, selecting regions for refinement, or identifying contact surfaces.

### Geometric measurements (OCC kernel only)

```python
mass = gmsh.model.occ.getMass(dim, tag)
# dim=1 → curve length, dim=2 → surface area, dim=3 → volume

cog = gmsh.model.occ.getCenterOfMass(dim, tag)
# Returns (x, y, z)
```

> [!note]
> `getMass` is named after the OCC convention — for geometry it returns length, area, or volume depending on dimension. Not to be confused with physical mass.

---

## File I/O

### Import

```python
gmsh.open(fileName)      # opens into a NEW model (replaces nothing, adds a model)
gmsh.merge(fileName)     # merges into the CURRENT model
```

The distinction matters:

- `open` — creates a fresh model from the file. Use for loading a standalone geometry or mesh.
- `merge` — adds the file's contents into the active model. Use for combining multiple geometries, or loading a mesh onto an existing geometry.

### Supported geometry formats

| Format | Extensions       | Kernel | Notes                              |
| ------ | ---------------- | ------ | ---------------------------------- |
| STEP   | `.step`, `.stp`  | OCC    | Industry standard, preferred       |
| IGES   | `.iges`, `.igs`  | OCC    | Legacy, less reliable than STEP    |
| BREP   | `.brep`          | OCC    | OpenCASCADE native format          |
| GEO    | `.geo`           | geo    | Gmsh's own scripting format        |

When importing CAD files (STEP/IGES/BREP), they go through the OCC kernel. The geometry may need healing:

```python
dimTags = gmsh.model.occ.importShapes("part.step")
gmsh.model.occ.healShapes()     # fix degenerate edges, small faces, etc.
gmsh.model.occ.synchronize()
```

### Supported mesh export formats

| Format   | Extension  | Notes                                       |
| -------- | ---------- | ------------------------------------------- |
| MSH4     | `.msh`     | Gmsh native, default                        |
| MSH2     | `.msh2`    | Legacy, needed by some tools                |
| VTK      | `.vtk`     | Legacy VTK format                           |
| VTU      | `.vtu`     | VTK XML unstructured grid                   |
| Abaqus   | `.inp`     | Also read by CalculiX                       |
| Nastran  | `.bdf`     | Bulk data format                            |
| UNV      | `.unv`     | IDEAS Universal                             |
| STL      | `.stl`     | Surface triangles only                      |
| CGNS     | `.cgns`    | CFD-oriented                                |
| MED      | `.med`     | Salome/Code_Aster                           |

```python
gmsh.write("model.msh")     # format inferred from extension
gmsh.write("model.vtu")
gmsh.write("model.inp")
```

### MSH format control

The MSH format version and content are controlled via options:

```python
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # force MSH2 for legacy tools
gmsh.option.setNumber("Mesh.SaveAll", 1)            # export all elements (ignore PG filter)
gmsh.option.setNumber("Mesh.Binary", 0)             # ASCII (1 for binary)
```

> [!important]
> For apeGmsh, file export is secondary — mesh data is extracted directly through the API. But `.msh` export is still useful for debugging (open in the Gmsh GUI) and for interoperability with tools that read MSH files.

---

## Options System

Gmsh uses a flat **key-value store** for all configuration. Options control geometry tolerances, mesh algorithms, export settings, visualization, and more.

```python
gmsh.option.setNumber(name, value)
gmsh.option.getNumber(name) → float

gmsh.option.setString(name, value)
gmsh.option.getString(name) → str
```

Option names follow the convention `Category.Name`:

```python
# Geometry
gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
gmsh.option.setNumber("Geometry.OCCSewFaces", 1)

# Mesh — global controls
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)
gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)         # global multiplier
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)   # elements per 2π

# Mesh — algorithm selection
gmsh.option.setNumber("Mesh.Algorithm", 6)       # 2D: Frontal-Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 10)    # 3D: HXT

# Mesh — element type control
gmsh.option.setNumber("Mesh.ElementOrder", 1)          # 1=linear, 2=quadratic
gmsh.option.setNumber("Mesh.RecombineAll", 1)          # quads/hexes everywhere
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1) # 0=simple, 1=blossom

# Export
gmsh.option.setNumber("Mesh.SaveAll", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
```

### Scope

Options are **global** — they apply to the entire session, not per-model. Setting `Mesh.Algorithm = 6` affects whichever model you mesh next. There is no per-model option override.

This is another consequence of Gmsh's singleton architecture: one global state, one set of options.

### Per-entity overrides

For mesh control, some options can be overridden at the entity level through dedicated API calls:

```python
gmsh.model.mesh.setAlgorithm(dim, tag, algo)     # override algorithm for one entity
gmsh.model.mesh.setRecombine(dim, tag)            # enable recombine for one entity
gmsh.model.mesh.setSmoothing(dim, tag, val)       # smoothing passes for one entity
gmsh.model.mesh.setSizeAtParametricPoint(dim, tag, parametricCoord, size)
```

These take precedence over the global option for that specific entity.
